"""
Maple recommendation engine — v4 (hybrid content + collaborative)

Pipeline:
  1. Precomputed TF-IDF embeddings + content cosine-similarity matrix
  2. MovieLens 25M collaborative item-item similarity matrix
  3. Hybrid similarity: α·content + (1−α)·collaborative  (α = 0.6)
  4. Top-K neighbour cache per item (startup, O(1) lookup)
  5. Weighted user-preference vector (recency + signal weights from route layer)
  6. Candidate scoring via hybrid content+collaborative similarity
  7. MMR diversity re-ranking (inter-item similarity penalty)
  8. 80/20 exploit/explore split (adjacent-genre exploration)
  9. Final deduplication + merge across movie/TV stores
"""

import os
import ast
import json
import pickle
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process as rfprocess

OMDB_KEY      = 'trilogy'
OMDB_IMG_BASE = 'https://img.omdbapi.com/?apikey=' + OMDB_KEY + '&i='
TMDB_IMG      = 'https://image.tmdb.org/t/p/w500'

# ── Precomputed top-K cache size ──
TOP_K = 50

# ── Hybrid blending weight ──
HYBRID_ALPHA = 0.6

# ── Search scoring weights ──
W_TITLE_EXACT    = 100   # exact title match
W_TITLE_PREFIX   = 80    # title starts with query
W_TITLE_CONTAINS = 60    # title contains query
W_TITLE_FUZZY    = 45    # fuzzy title match
W_DIRECTOR       = 40    # director name match
W_CAST           = 35    # cast name match
W_GENRE          = 20    # genre match
W_OVERVIEW       = 10    # description match


class ContentStore:
    """Holds data + similarity matrices + TF-IDF matrix + precomputed caches."""

    def __init__(self, csv_path, sim_path, content_type,
                 tfidf_matrix_path=None, collab_sim_path=None):
        self.content_type  = content_type
        self.ready         = False
        self.tfidf_matrix  = None
        self.collab_sim    = None   # collaborative item-item similarity (MovieLens)
        self.hybrid_sim    = None   # blended content + collaborative
        self.top_k_cache   = None   # ndarray (n_items, TOP_K) of positions
        self.top_k_scores  = None   # ndarray (n_items, TOP_K) of similarity scores
        self._genre_index  = {}     # genre → set of positions

        if not os.path.exists(csv_path) or not os.path.exists(sim_path):
            print(f"[ML] Missing files for {content_type}")
            self.df  = pd.DataFrame()
            self.sim = None
            return

        self.df = pd.read_csv(csv_path)
        with open(sim_path, 'rb') as f:
            self.sim = pickle.load(f)

        self.positional_index = {int(row_id): pos for pos, row_id in enumerate(self.df['id'])}
        self.ready = True
        print(f"[ML] {content_type}: {len(self.df)} items loaded.")

        # Load sparse TF-IDF matrix
        if tfidf_matrix_path and os.path.exists(tfidf_matrix_path):
            with open(tfidf_matrix_path, 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
            # Ensure it's sparse CSR for efficient row slicing + matrix ops
            if not sparse.issparse(self.tfidf_matrix):
                self.tfidf_matrix = sparse.csr_matrix(self.tfidf_matrix)
            elif not sparse.isspmatrix_csr(self.tfidf_matrix):
                self.tfidf_matrix = self.tfidf_matrix.tocsr()
            print(f"[ML] {content_type}: TF-IDF matrix loaded ({self.tfidf_matrix.shape}).")

        # Load collaborative similarity (MovieLens 25M)
        if collab_sim_path and os.path.exists(collab_sim_path):
            with open(collab_sim_path, 'rb') as f:
                self.collab_sim = pickle.load(f)
            print(f"[ML] {content_type}: Collaborative similarity loaded ({self.collab_sim.shape}).")

        # Build hybrid similarity: α·content + (1−α)·collaborative
        self._build_hybrid_sim()

        # Build precomputed caches (uses hybrid_sim if available)
        self._build_top_k_cache()
        self._build_genre_index()
        self._build_search_index()

    def _build_hybrid_sim(self):
        """Blend content and collaborative similarity matrices."""
        if self.sim is None:
            return
        if self.collab_sim is not None and self.collab_sim.shape == self.sim.shape:
            alpha = HYBRID_ALPHA
            self.hybrid_sim = alpha * self.sim + (1 - alpha) * self.collab_sim
            print(f"[ML] {self.content_type}: Hybrid similarity built "
                  f"(α={alpha} content + {1-alpha:.1f} collaborative).")
        else:
            # Fallback: content-only
            self.hybrid_sim = self.sim
            if self.collab_sim is not None:
                print(f"[ML] {self.content_type}: Shape mismatch — "
                      f"content {self.sim.shape} vs collab {self.collab_sim.shape}, "
                      f"using content-only.")

    # ------------------------------------------------------------------
    # Startup caches
    # ------------------------------------------------------------------

    def _build_top_k_cache(self):
        """Precompute top-K neighbours using hybrid similarity (or content-only fallback)."""
        sim = self.hybrid_sim if self.hybrid_sim is not None else self.sim
        if sim is None:
            return
        n = len(self.df)
        k = min(TOP_K, n - 1)
        print(f"[ML] {self.content_type}: building top-{k} neighbour cache…")

        # argpartition is O(n) per row — much faster than full argsort
        top_indices = np.argpartition(sim, -k - 1, axis=1)[:, -k - 1:]
        rows = np.arange(n)[:, None]
        top_scores = sim[rows, top_indices]
        sorted_order = np.argsort(-top_scores, axis=1)
        top_indices = np.take_along_axis(top_indices, sorted_order, axis=1)
        top_scores  = np.take_along_axis(top_scores, sorted_order, axis=1)

        # Remove self (position 0 after sort is always self with score=1.0)
        self.top_k_cache  = top_indices[:, 1:k + 1]
        self.top_k_scores = top_scores[:, 1:k + 1]
        print(f"[ML] {self.content_type}: top-K cache built ({self.top_k_cache.shape}).")

    def _build_genre_index(self):
        """Map each genre → set of positional indices for fast explore lookups."""
        self._genre_index = {}
        for pos, g_str in enumerate(self.df['genres_display'].fillna('')):
            for g in str(g_str).split(','):
                g = g.strip()
                if g:
                    self._genre_index.setdefault(g, set()).add(pos)

    def _build_search_index(self):
        """Build inverted indexes for O(1) lookup instead of full-scan search."""
        self._title_index = {}       # lowercase title → pos
        self._title_list  = []       # [(lowercase_title, pos)] for fuzzy matching
        self._cast_index  = defaultdict(set)   # lowercase name → set of pos
        self._dir_index   = defaultdict(set)   # lowercase name → set of pos
        self._cast_names  = []       # unique cast names for autocomplete
        self._dir_names   = []       # unique director names for autocomplete

        cast_set = set()
        dir_set  = set()

        for pos in range(len(self.df)):
            row = self.df.iloc[pos]

            # Title index
            title_lower = str(row['title']).lower().strip()
            self._title_index[title_lower] = pos
            self._title_list.append((title_lower, pos))

            # Cast index — split space-separated names into individual full names
            cast_raw = str(row.get('cast', '') or '')
            if cast_raw and cast_raw != 'nan':
                # cast_json has proper names; use it if available
                cj = str(row.get('cast_json', '') or '')
                if cj and cj != 'nan':
                    try:
                        for c in ast.literal_eval(cj):
                            name = c.get('name', '').strip()
                            if name:
                                self._cast_index[name.lower()].add(pos)
                                cast_set.add(name)
                    except Exception:
                        pass
                else:
                    # Fallback: space-separated names are unreliable for splitting,
                    # but we index the whole blob for substring matching
                    pass

            # Director index
            dir_raw = str(row.get('director', '') or '')
            if dir_raw and dir_raw != 'nan':
                for d in dir_raw.split(','):
                    d = d.strip()
                    if d:
                        self._dir_index[d.lower()].add(pos)
                        dir_set.add(d)

        self._cast_names = sorted(cast_set)
        self._dir_names  = sorted(dir_set)
        print(f"[ML] {self.content_type}: search index built — "
              f"{len(self._title_list)} titles, "
              f"{len(self._cast_names)} cast, "
              f"{len(self._dir_names)} directors.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_dict(self, row):
        poster = str(row.get('poster', '') or '')
        if not poster:
            imdb_id_raw = row.get('imdb_id', 0)
            if isinstance(imdb_id_raw, str) and imdb_id_raw.startswith('tt'):
                poster = OMDB_IMG_BASE + imdb_id_raw
            else:
                imdb_id = int(imdb_id_raw or 0)
                if imdb_id > 0:
                    poster = OMDB_IMG_BASE + f'tt{imdb_id:07d}'

        # Cast / director fields
        cast_raw = str(row.get('cast', '') or '')
        cast_display = cast_raw if cast_raw != 'nan' else ''
        director_raw = str(row.get('director', '') or '')
        director = director_raw if director_raw != 'nan' else ''

        # Convert Python repr cast_json → proper JSON for frontend
        cast_json_raw = str(row.get('cast_json', '') or '')
        cast_json = ''
        if cast_json_raw and cast_json_raw != 'nan':
            try:
                cast_json = json.dumps(ast.literal_eval(cast_json_raw))
            except Exception:
                cast_json = ''

        d = {
            'id':           int(row['id']),
            'title':        str(row['title']),
            'overview':     str(row.get('overview', '')),
            'genres':       str(row.get('genres_display', '')),
            'poster':       poster,
            'rating':       round(float(row.get('vote_average', 0)), 1),
            'year':         int(row.get('release_year', 0)),
            'content_type': self.content_type,
            'cast':         cast_display,
            'director':     director,
            'cast_json':    cast_json,
        }
        if self.content_type == 'tv':
            d['seasons']  = int(row.get('number_of_seasons', 0))
            d['episodes'] = int(row.get('number_of_episodes', 0))
            d['networks'] = str(row.get('networks', ''))
            d['status']   = str(row.get('status', ''))
        return d

    def _compute_user_vector_and_scores(self, item_ids, weights=None, exclude_ids=None):
        """
        Build weighted user-preference vector → compute hybrid similarity
        against all items → return (scored_list, user_vec_or_None).

        Hybrid scoring: α·content_score + (1−α)·collaborative_score
        scored_list: [(pos, score), ...] sorted desc, excluding exclude_ids.
        """
        pos_weights = []
        for idx, m in enumerate(item_ids):
            mid = int(m)
            if mid in self.positional_index:
                w = weights[idx] if weights else 1.0
                pos_weights.append((self.positional_index[mid], w))

        if not pos_weights:
            return [], None

        positions = [p for p, _ in pos_weights]
        w_array   = np.array([w for _, w in pos_weights], dtype=np.float64)
        w_sum = w_array.sum()
        if w_sum > 0:
            w_array /= w_sum

        exclude = {int(m) for m in item_ids}
        if exclude_ids:
            exclude |= {int(m) for m in exclude_ids}

        # Content-based scores
        if self.tfidf_matrix is not None:
            vecs = self.tfidf_matrix[positions]
            user_vec = vecs.T.dot(w_array)
            user_vec = user_vec.reshape(1, -1)
            content_scores = cosine_similarity(user_vec, self.tfidf_matrix)[0]
        else:
            rows = self.sim[positions]
            content_scores = w_array @ rows
            user_vec = None

        # Collaborative scores (if available)
        if self.collab_sim is not None:
            collab_rows = self.collab_sim[positions]
            collab_scores = w_array @ collab_rows
            # Hybrid blend
            alpha = HYBRID_ALPHA
            scores = alpha * content_scores + (1 - alpha) * collab_scores
        else:
            scores = content_scores

        scored = [
            (pos, float(scores[pos]))
            for pos in range(len(self.df))
            if int(self.df.iloc[pos]['id']) not in exclude
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored, (user_vec if self.tfidf_matrix is not None else None)

    # ------------------------------------------------------------------
    # Diversity: Maximal Marginal Relevance (MMR)
    # ------------------------------------------------------------------

    def _mmr_rerank(self, scored, n, lam=0.65):
        """
        Maximal Marginal Relevance re-ranking.

        MMR(d) = λ · sim(d, user) − (1−λ) · max_{d_j ∈ S} sim(d, d_j)

        λ = 0.65 → 65% relevance, 35% diversity penalty.
        Only the top candidate_pool (3×n) is considered to keep it fast.

        Uses the precomputed similarity matrix for inter-item similarity.
        """
        if len(scored) <= n:
            return [self._row_to_dict(self.df.iloc[pos]) for pos, _ in scored]

        pool_size = min(len(scored), n * 3)
        pool      = scored[:pool_size]

        selected_positions = []
        selected_set       = set()
        remaining          = list(range(len(pool)))

        # Always pick the top-1 by pure relevance
        best = remaining[0]
        selected_positions.append(pool[best][0])
        selected_set.add(best)
        remaining.remove(best)

        for _ in range(n - 1):
            if not remaining:
                break

            best_idx  = -1
            best_mmr  = -float('inf')

            for r_idx in remaining:
                cand_pos, cand_score = pool[r_idx]

                # Max similarity to any already-selected item (use hybrid if available)
                max_sim_to_selected = 0.0
                sim_mat = self.hybrid_sim if self.hybrid_sim is not None else self.sim
                for sel_pos in selected_positions:
                    if sim_mat is not None:
                        pair_sim = float(sim_mat[cand_pos, sel_pos])
                    else:
                        pair_sim = 0.0
                    if pair_sim > max_sim_to_selected:
                        max_sim_to_selected = pair_sim

                mmr = lam * cand_score - (1.0 - lam) * max_sim_to_selected

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = r_idx

            if best_idx < 0:
                break
            selected_positions.append(pool[best_idx][0])
            selected_set.add(best_idx)
            remaining.remove(best_idx)

        return [self._row_to_dict(self.df.iloc[pos]) for pos in selected_positions]

    # ------------------------------------------------------------------
    # Exploration: find items from adjacent genres
    # ------------------------------------------------------------------

    def _get_explore_candidates(self, item_ids, exclude, n=10):
        """
        Find exploration candidates — items from genres *adjacent* to the user's
        taste but not in the top-similarity bucket.

        Strategy:
          1. Collect all genres the user has interacted with.
          2. For each genre, find its "sibling" genres (genres that co-occur
             on the same items).
          3. Gather items from sibling genres that the user hasn't seen.
          4. Rank by weighted_rating (quality gate) and sample.
        """
        if not self.ready:
            return []

        # Collect user's genres
        user_genres = set()
        for mid in item_ids:
            mid = int(mid)
            if mid in self.positional_index:
                pos = self.positional_index[mid]
                for g in str(self.df.iloc[pos].get('genres_display', '')).split(','):
                    g = g.strip()
                    if g:
                        user_genres.add(g)

        if not user_genres:
            return []

        # Find sibling genres — genres that share items with user's genres
        sibling_genres = set()
        for g in user_genres:
            for pos in list(self._genre_index.get(g, set()))[:200]:
                for g2 in str(self.df.iloc[pos].get('genres_display', '')).split(','):
                    g2 = g2.strip()
                    if g2 and g2 not in user_genres:
                        sibling_genres.add(g2)

        if not sibling_genres:
            return []

        # Collect candidate positions from sibling genres
        exclude_ids = {int(m) for m in item_ids} | (exclude or set())
        candidate_positions = set()
        for g in sibling_genres:
            candidate_positions |= self._genre_index.get(g, set())

        # Remove items the user has already interacted with
        candidates = [
            pos for pos in candidate_positions
            if int(self.df.iloc[pos]['id']) not in exclude_ids
        ]

        if not candidates:
            return []

        # Rank by quality (weighted_rating or vote_average)
        sort_col = 'weighted_rating' if 'weighted_rating' in self.df.columns else 'vote_average'
        rated = [(pos, float(self.df.iloc[pos].get(sort_col, 0))) for pos in candidates]
        rated.sort(key=lambda x: x[1], reverse=True)

        # Take top pool and sample n from it for variety
        pool = rated[:max(n * 4, 40)]
        chosen = random.sample(pool, min(n, len(pool)))
        return [self._row_to_dict(self.df.iloc[pos]) for pos, _ in chosen]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_item(self, item_id):
        item_id = int(item_id)
        match   = self.df[self.df['id'] == item_id]
        if match.empty:
            return None
        return self._row_to_dict(match.iloc[0])

    def search(self, query, limit=24):
        """
        Weighted search pipeline:
          1. Exact / prefix / contains title match (indexed, O(n) titles)
          2. Fuzzy title match via rapidfuzz (handles typos)
          3. Cast name match (indexed, O(1) per name)
          4. Director name match (indexed, O(1) per name)
          5. Genre match
          6. Overview substring match (only if few results so far)
          7. Score and rank all matches, return top-limit
        """
        if self.df.empty:
            return []
        q = query.lower().strip()
        if not q:
            return []

        scores = defaultdict(float)  # pos → cumulative score
        rating_col = 'weighted_rating' if 'weighted_rating' in self.df.columns else 'vote_average'

        # 1) Title matching — exact / prefix / contains
        for title_lower, pos in self._title_list:
            if title_lower == q:
                scores[pos] += W_TITLE_EXACT
            elif title_lower.startswith(q):
                scores[pos] += W_TITLE_PREFIX
            elif q in title_lower:
                scores[pos] += W_TITLE_CONTAINS

        # 2) Fuzzy title match (catches typos like "intersteller" → "interstellar")
        #    Only for single-word queries; multi-word queries are usually person
        #    names or phrases, where WRatio produces false positives like
        #    "hugh jackman" fuzzy-matching "Pan" via shared letters.
        q_words = q.split()
        if len(q) >= 3 and len(q_words) == 1:
            fuzzy_results = rfprocess.extract(
                q, [t for t, _ in self._title_list],
                scorer=fuzz.WRatio, limit=30, score_cutoff=70
            )
            for matched_title, score, idx in fuzzy_results:
                pos = self._title_list[idx][1]
                # Penalise very short titles that match purely by substring noise
                len_ratio = min(len(matched_title), len(q)) / max(len(matched_title), len(q))
                fuzzy_score = W_TITLE_FUZZY * (score / 100) * max(len_ratio, 0.5)
                # Only add if not already found by exact/prefix/contains
                if scores[pos] < W_TITLE_CONTAINS:
                    scores[pos] = max(scores[pos], fuzzy_score)

        # 3) Cast match — check if query matches any indexed cast name
        #    Detect person-intent: if query is an exact or near-exact cast/director
        #    name, boost those matches so they dominate over stray overview hits.
        person_boost = 0
        for cast_lower in self._cast_index:
            if cast_lower == q or q == cast_lower:
                person_boost = 30  # exact person match — big boost
                break
            elif q in cast_lower and len(q) >= 6:
                person_boost = max(person_boost, 15)

        for cast_lower, positions in self._cast_index.items():
            if q in cast_lower:
                for pos in positions:
                    scores[pos] += W_CAST + person_boost

        # 4) Director match
        for dir_lower in self._dir_index:
            if dir_lower == q or q == dir_lower:
                person_boost = max(person_boost, 30)
                break
            elif q in dir_lower and len(q) >= 6:
                person_boost = max(person_boost, 15)

        for dir_lower, positions in self._dir_index.items():
            if q in dir_lower:
                for pos in positions:
                    scores[pos] += W_DIRECTOR + person_boost

        # 5) Genre match
        for genre, positions in self._genre_index.items():
            if q in genre.lower():
                for pos in positions:
                    scores[pos] += W_GENRE

        # 6) Overview / description match — adds signal to all items
        overview_col = self.df['overview'].fillna('').str.lower()
        mask = overview_col.str.contains(q, na=False, regex=False)
        for pos in mask[mask].index:
            scores[pos] += W_OVERVIEW

        if not scores:
            return []

        # Rank by score, break ties with rating
        ranked = sorted(
            scores.items(),
            key=lambda x: (x[1], float(self.df.iloc[x[0]].get(rating_col, 0))),
            reverse=True
        )

        results = []
        for pos, sc in ranked[:limit]:
            d = self._row_to_dict(self.df.iloc[pos])
            d['_score'] = sc
            results.append(d)
        return results

    def autocomplete(self, query, limit=8):
        """
        Fast autocomplete suggestions for the search box.
        Returns a list of {text, type, id?} suggestions.
        """
        if self.df.empty or not query:
            return []
        q = query.lower().strip()
        if len(q) < 2:
            return []

        suggestions = []

        # Title prefix matches (fast, indexed)
        for title_lower, pos in self._title_list:
            if title_lower.startswith(q):
                row = self.df.iloc[pos]
                suggestions.append({
                    'text': str(row['title']),
                    'type': self.content_type,
                    'id':   int(row['id']),
                    'year': int(row.get('release_year', 0)),
                    'poster': str(row.get('poster', '') or ''),
                    'rating': round(float(row.get('vote_average', 0)), 1),
                    '_score': 100,
                })
            if len(suggestions) >= limit * 2:
                break

        # Title contains (if not enough prefix matches)
        if len(suggestions) < limit:
            for title_lower, pos in self._title_list:
                if q in title_lower and not title_lower.startswith(q):
                    row = self.df.iloc[pos]
                    suggestions.append({
                        'text': str(row['title']),
                        'type': self.content_type,
                        'id':   int(row['id']),
                        'year': int(row.get('release_year', 0)),
                        'poster': str(row.get('poster', '') or ''),
                        'rating': round(float(row.get('vote_average', 0)), 1),
                        '_score': 80,
                    })
                if len(suggestions) >= limit * 2:
                    break

        # Cast name matches
        for name in self._cast_names:
            if q in name.lower():
                count = len(self._cast_index.get(name.lower(), set()))
                suggestions.append({
                    'text': name,
                    'type': 'person',
                    'sub':  f'{count} title{"s" if count != 1 else ""}',
                    '_score': 70,
                })
            if len(suggestions) >= limit * 3:
                break

        # Director name matches
        for name in self._dir_names:
            if q in name.lower():
                count = len(self._dir_index.get(name.lower(), set()))
                suggestions.append({
                    'text': name,
                    'type': 'director',
                    'sub':  f'{count} title{"s" if count != 1 else ""}',
                    '_score': 70,
                })
            if len(suggestions) >= limit * 3:
                break

        # Fuzzy title matches if still short on title suggestions
        title_count = len([s for s in suggestions if s['type'] != 'person' and s['type'] != 'director'])
        if title_count < 3 and len(q) >= 3:
            fuzzy_results = rfprocess.extract(
                q, [t for t, _ in self._title_list],
                scorer=fuzz.WRatio, limit=8, score_cutoff=65
            )
            existing_ids = {s.get('id') for s in suggestions if 'id' in s}
            for matched_title, score, idx in fuzzy_results:
                # Skip very short titles that are noise matches
                if len(matched_title) < 3:
                    continue
                len_ratio = min(len(matched_title), len(q)) / max(len(matched_title), len(q))
                adj_score = score * max(len_ratio, 0.5)
                if adj_score < 50:
                    continue
                pos = self._title_list[idx][1]
                row = self.df.iloc[pos]
                rid = int(row['id'])
                if rid not in existing_ids:
                    suggestions.append({
                        'text': str(row['title']),
                        'type': self.content_type,
                        'id':   rid,
                        'year': int(row.get('release_year', 0)),
                        'poster': str(row.get('poster', '') or ''),
                        'rating': round(float(row.get('vote_average', 0)), 1),
                        '_score': adj_score * 0.6,
                    })

        # Sort by score, deduplicate by text
        suggestions.sort(key=lambda x: x['_score'], reverse=True)
        seen = set()
        deduped = []
        for s in suggestions:
            key = s['text'].lower()
            if key not in seen:
                seen.add(key)
                s.pop('_score', None)
                deduped.append(s)
            if len(deduped) >= limit:
                break

        return deduped

    def get_page(self, page=1, per_page=24, genre=None):
        df = self.df
        if genre:
            df = df[df['genres_display'].str.contains(genre, case=False, na=False)]
        total = len(df)
        start = (page - 1) * per_page
        chunk = df.iloc[start:start + per_page]
        return {
            'items': [self._row_to_dict(r) for _, r in chunk.iterrows()],
            'total': total,
            'page':  page,
            'pages': (total + per_page - 1) // per_page,
        }

    def get_similar(self, item_id, n=12):
        """O(1) lookup using precomputed top-K cache."""
        if not self.ready:
            return []
        item_id = int(item_id)
        if item_id not in self.positional_index:
            return []
        pos = self.positional_index[item_id]
        if self.top_k_cache is not None:
            k = min(n, self.top_k_cache.shape[1])
            neighbours = self.top_k_cache[pos, :k]
            return [self._row_to_dict(self.df.iloc[int(i)]) for i in neighbours]
        # Fallback to full sort
        scores = sorted(enumerate(self.sim[pos]), key=lambda x: x[1], reverse=True)[1:n + 1]
        return [self._row_to_dict(self.df.iloc[i]) for i, _ in scores]

    def get_personalized(self, item_ids, n=12, exclude_ids=None, weights=None):
        """
        Full recommendation pipeline for one content store:
          1. Build weighted user-preference vector
          2. Score all candidates
          3. 80/20 exploit/explore split
          4. MMR diversity re-ranking on the exploit portion
          5. Merge exploit + explore
        """
        if not self.ready or not item_ids:
            return self.top_rated(n)

        scored, _ = self._compute_user_vector_and_scores(item_ids, weights, exclude_ids)

        if not scored:
            return self.top_rated(n)

        # ── 80/20 exploit / explore split ──
        n_exploit = max(1, int(n * 0.8))
        n_explore = n - n_exploit

        # Exploit: MMR-diversified top candidates
        exploit_results = self._mmr_rerank(scored, n_exploit, lam=0.65)

        # Explore: adjacent-genre items the user hasn't seen
        exclude_all = {int(m) for m in item_ids}
        if exclude_ids:
            exclude_all |= {int(m) for m in exclude_ids}
        # Also exclude exploit results from explore pool
        exploit_ids = {r['id'] for r in exploit_results}
        exclude_all |= exploit_ids

        explore_results = self._get_explore_candidates(item_ids, exclude_all, n=n_explore)

        # Merge: exploit first, explore appended
        combined = exploit_results + explore_results

        # Deduplicate by ID (safety)
        seen = set()
        deduped = []
        for item in combined:
            if item['id'] not in seen:
                seen.add(item['id'])
                deduped.append(item)

        return deduped[:n]

    def get_surprise(self, item_ids, exclude_ids=None, weights=None):
        """Pick a single random item, weighted by personalised relevance score."""
        if not self.ready or not item_ids:
            sort_col = 'weighted_rating' if 'weighted_rating' in self.df.columns else 'vote_average'
            pool = self.df.nlargest(100, sort_col)
            row  = pool.iloc[random.randint(0, len(pool) - 1)]
            return self._row_to_dict(row)

        scored, _ = self._compute_user_vector_and_scores(item_ids, weights, exclude_ids)

        if not scored:
            sort_col = 'weighted_rating' if 'weighted_rating' in self.df.columns else 'vote_average'
            pool = self.df.nlargest(100, sort_col)
            row  = pool.iloc[random.randint(0, len(pool) - 1)]
            return self._row_to_dict(row)

        # Sample from top-50, weighted by score
        pool = scored[:50]
        pool_scores = np.array([s for _, s in pool])
        pool_scores = pool_scores - pool_scores.min() + 0.01
        probs       = pool_scores / pool_scores.sum()
        idx         = np.random.choice(len(pool), p=probs)
        return self._row_to_dict(self.df.iloc[pool[idx][0]])

    def top_rated(self, n=12):
        sort_col = 'weighted_rating' if 'weighted_rating' in self.df.columns else 'vote_average'
        top = self.df.nlargest(n, sort_col)
        return [self._row_to_dict(r) for _, r in top.iterrows()]

    def get_genres(self):
        return sorted(self._genre_index.keys())


# ---------------------------------------------------------------------------
# Top-level engine — orchestrates movie + TV stores
# ---------------------------------------------------------------------------

class RecommendationEngine:
    def __init__(self, movies_path, movies_sim_path, tv_path, tv_sim_path,
                 movies_tfidf_path=None, tv_tfidf_path=None,
                 movies_collab_sim_path=None, tv_collab_sim_path=None):
        print("[ML] Loading movies...")
        self.movies = ContentStore(
            movies_path, movies_sim_path, 'movie',
            tfidf_matrix_path=movies_tfidf_path,
            collab_sim_path=movies_collab_sim_path,
        )
        print("[ML] Loading TV shows...")
        self.tv = ContentStore(
            tv_path, tv_sim_path, 'tv',
            tfidf_matrix_path=tv_tfidf_path,
            collab_sim_path=tv_collab_sim_path,
        )

        # LRU cache for user vectors (cleared on each app restart)
        self._user_cache = {}

    def _store(self, content_type):
        return self.tv if content_type == 'tv' else self.movies

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_item(self, item_id, content_type='movie'):
        return self._store(content_type).get_item(item_id)

    def search(self, query, content_type=None, limit=24):
        if content_type == 'movie':
            results = self.movies.search(query, limit)
            for r in results:
                r.pop('_score', None)
            return results
        if content_type == 'tv':
            results = self.tv.search(query, limit)
            for r in results:
                r.pop('_score', None)
            return results
        # Combined: merge both stores sorted by score
        m = self.movies.search(query, limit)
        t = self.tv.search(query, limit)
        combined = sorted(m + t, key=lambda x: (x.get('_score', 0), x.get('rating', 0)), reverse=True)
        seen = set()
        deduped = []
        for item in combined:
            key = item['title'].lower().strip()
            if key not in seen:
                seen.add(key)
                item.pop('_score', None)
                deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped

    def autocomplete(self, query, content_type=None, limit=8):
        if content_type == 'movie':
            return self.movies.autocomplete(query, limit)
        if content_type == 'tv':
            return self.tv.autocomplete(query, limit)
        m = self.movies.autocomplete(query, limit)
        t = self.tv.autocomplete(query, limit)
        # Merge and deduplicate
        combined = []
        seen = set()
        for s in sorted(m + t, key=lambda x: x.get('_score', 0), reverse=True):
            key = s['text'].lower()
            if key not in seen:
                seen.add(key)
                s.pop('_score', None)
                combined.append(s)
            if len(combined) >= limit:
                break
        return combined

    def get_all(self, page=1, per_page=24, genre=None, content_type=None):
        if content_type == 'tv':
            r = self.tv.get_page(page, per_page, genre)
            return {'movies': r['items'], 'total': r['total'], 'page': r['page'], 'pages': r['pages']}
        if content_type == 'movie':
            r = self.movies.get_page(page, per_page, genre)
            return {'movies': r['items'], 'total': r['total'], 'page': r['page'], 'pages': r['pages']}
        half = per_page // 2
        mr = self.movies.get_page(page, half, genre)
        tr = self.tv.get_page(page, half, genre)
        combined = []
        for i in range(max(len(mr['items']), len(tr['items']))):
            if i < len(mr['items']): combined.append(mr['items'][i])
            if i < len(tr['items']): combined.append(tr['items'][i])
        total = mr['total'] + tr['total']
        return {'movies': combined, 'total': total, 'page': page, 'pages': (total + per_page - 1) // per_page}

    def get_similar(self, item_id, content_type='movie', n=12):
        return self._store(content_type).get_similar(item_id, n)

    def get_personalized(self, movie_ids, tv_ids, n=12,
                          exclude_movie_ids=None, exclude_tv_ids=None,
                          movie_weights=None, tv_weights=None):
        m_recs = self.movies.get_personalized(
            movie_ids, n,
            exclude_ids=exclude_movie_ids,
            weights=movie_weights,
        ) if movie_ids else []
        t_recs = self.tv.get_personalized(
            tv_ids, n,
            exclude_ids=exclude_tv_ids,
            weights=tv_weights,
        ) if tv_ids else []

        if not m_recs and not t_recs:
            return self.movies.top_rated(n)

        # Deduplicate by title across movie + TV
        seen_titles = set()
        combined = []
        for i in range(max(len(m_recs), len(t_recs))):
            for lst in (m_recs, t_recs):
                if i < len(lst):
                    key = lst[i]['title'].lower().strip()
                    if key not in seen_titles:
                        seen_titles.add(key)
                        combined.append(lst[i])
        return combined[:n]

    def get_surprise(self, movie_ids, tv_ids,
                     exclude_movie_ids=None, exclude_tv_ids=None,
                     movie_weights=None, tv_weights=None):
        m_count = len(movie_ids) if movie_ids else 0
        t_count = len(tv_ids) if tv_ids else 0
        total   = m_count + t_count

        if total == 0:
            store = random.choice([self.movies, self.tv])
            return store.get_surprise([], exclude_ids=None)

        pick_movie = random.random() < (m_count / total if total else 0.5)
        if pick_movie and movie_ids:
            return self.movies.get_surprise(
                movie_ids, exclude_ids=exclude_movie_ids, weights=movie_weights)
        elif tv_ids:
            return self.tv.get_surprise(
                tv_ids, exclude_ids=exclude_tv_ids, weights=tv_weights)
        else:
            return self.movies.get_surprise(
                movie_ids, exclude_ids=exclude_movie_ids, weights=movie_weights)

    def top_rated(self, content_type=None, n=12):
        if content_type == 'tv':
            return self.tv.top_rated(n)
        if content_type == 'movie':
            return self.movies.top_rated(n)
        m = self.movies.top_rated(n // 2)
        t = self.tv.top_rated(n // 2)
        combined = []
        for i in range(max(len(m), len(t))):
            if i < len(m): combined.append(m[i])
            if i < len(t): combined.append(t[i])
        return combined

    def get_genres(self, content_type=None):
        if content_type == 'tv':
            return self.tv.get_genres()
        if content_type == 'movie':
            return self.movies.get_genres()
        return sorted(set(self.movies.get_genres()) | set(self.tv.get_genres()))

    def get_genre_stats(self, movie_ids, tv_ids):
        from collections import Counter
        counts = Counter()
        for mid in set(movie_ids):
            row = self.movies.df[self.movies.df['id'] == mid]
            if not row.empty:
                for g in str(row.iloc[0].get('genres_display', '')).split(','):
                    g = g.strip()
                    if g:
                        counts[g] += 1
        for tid in set(tv_ids):
            row = self.tv.df[self.tv.df['id'] == tid]
            if not row.empty:
                for g in str(row.iloc[0].get('genres_display', '')).split(','):
                    g = g.strip()
                    if g:
                        counts[g] += 1
        if not counts:
            return []
        total = sum(counts.values())
        return [{'genre': g, 'count': c, 'pct': round(c * 100 / total)}
                for g, c in counts.most_common(8)]
