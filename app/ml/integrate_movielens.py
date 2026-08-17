"""
Maple — MovieLens 25M Integration Script

This script integrates the MovieLens 25M dataset to:
  1. Enrich movie features with user tags + genome tags → better TF-IDF
  2. Build an item-item collaborative similarity matrix from 25M ratings
  3. Merge improved ratings data (25M crowd-sourced ratings)

Run from project root:
    python -m app.ml.integrate_movielens

Outputs:
    data/processed/processed_movies2.csv          — enriched CSV (updated in-place)
    data/processed/collab_similarity_movies.pkl    — item-item collaborative similarity
    data/processed/tfidf_matrix2.pkl               — rebuilt TF-IDF with tags
    data/processed/similarity_matrix2.pkl          — rebuilt content similarity
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Paths
BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROC_DIR  = os.path.join(BASE_DIR, 'data', 'processed')
ML_DIR    = os.path.join(BASE_DIR, 'data', 'ml-25m')

MOVIES_CSV      = os.path.join(PROC_DIR, 'processed_movies2.csv')
SIM_PKL         = os.path.join(PROC_DIR, 'similarity_matrix2.pkl')
TFIDF_PKL       = os.path.join(PROC_DIR, 'tfidf_matrix2.pkl')
TFIDF_VEC_PKL   = os.path.join(PROC_DIR, 'tfidf_vectorizer2.pkl')
COLLAB_SIM_PKL  = os.path.join(PROC_DIR, 'collab_similarity_movies.pkl')

# MovieLens files
ML_LINKS    = os.path.join(ML_DIR, 'links.csv')
ML_RATINGS  = os.path.join(ML_DIR, 'ratings.csv')
ML_TAGS     = os.path.join(ML_DIR, 'tags.csv')
ML_GENOME_S = os.path.join(ML_DIR, 'genome-scores.csv')
ML_GENOME_T = os.path.join(ML_DIR, 'genome-tags.csv')


def load_id_mapping():
    """Build MovieLens movieId ↔ TMDB ID mapping."""
    print("[1/6] Loading ID mapping (links.csv)...")
    links = pd.read_csv(ML_LINKS)
    links = links.dropna(subset=['tmdbId']).copy()
    links['tmdbId'] = links['tmdbId'].astype(int)
    # ml_to_tmdb: MovieLens ID → TMDB ID
    ml_to_tmdb = dict(zip(links['movieId'], links['tmdbId']))
    tmdb_to_ml = dict(zip(links['tmdbId'], links['movieId']))
    print(f"    {len(ml_to_tmdb)} MovieLens→TMDB mappings loaded.")
    return ml_to_tmdb, tmdb_to_ml


def enrich_with_tags(movies_df, tmdb_to_ml):
    """Add user tags + genome tags to each movie for TF-IDF enrichment."""
    print("[2/6] Enriching with tags...")

    our_tmdb_ids = set(movies_df['id'].astype(int))
    our_ml_ids = {}
    for tmdb_id in our_tmdb_ids:
        if tmdb_id in tmdb_to_ml:
            our_ml_ids[tmdb_to_ml[tmdb_id]] = tmdb_id

    # ── User tags ──
    print("    Loading user tags...")
    tags_df = pd.read_csv(ML_TAGS, usecols=['movieId', 'tag'])
    tags_df = tags_df[tags_df['movieId'].isin(our_ml_ids)].copy()
    tags_df['tag'] = tags_df['tag'].astype(str).str.lower().str.strip()
    # Aggregate tags per movie, deduplicate
    user_tags = (tags_df.groupby('movieId')['tag']
                 .apply(lambda x: ' '.join(set(x)))
                 .to_dict())
    print(f"    {len(user_tags)} movies enriched with user tags.")

    # ── Genome tags (top-scoring tags per movie) ──
    print("    Loading genome tags (this may take a minute)...")
    genome_tags = pd.read_csv(ML_GENOME_T)
    tag_names = dict(zip(genome_tags['tagId'], genome_tags['tag']))

    # Load genome scores in chunks to avoid memory + index issues
    genome_top = {}
    ml_id_set = set(our_ml_ids.keys())
    chunks = pd.read_csv(ML_GENOME_S, chunksize=2_000_000)
    collected = []
    for chunk in chunks:
        sub = chunk[chunk['movieId'].isin(ml_id_set) & (chunk['relevance'] > 0.5)]
        if len(sub) > 0:
            collected.append(sub[['movieId', 'tagId', 'relevance']].copy())

    if collected:
        genome_filtered = pd.concat(collected, ignore_index=True)
        genome_filtered = genome_filtered.assign(tag=genome_filtered['tagId'].map(tag_names))
        genome_filtered = genome_filtered.dropna(subset=['tag']).reset_index(drop=True)
        # Group and aggregate manually to avoid sort_values pandas bug
        genome_filtered['tag'] = genome_filtered['tag'].str.lower().str.strip()
        grouped = genome_filtered.groupby('movieId')
        for ml_id, group in grouped:
            top = group.nlargest(15, 'relevance')
            genome_top[ml_id] = ' '.join(top['tag'].tolist())
    print(f"    {len(genome_top)} movies enriched with genome tags.")

    # ── Merge tags into dataframe ──
    tag_col = []
    for _, row in movies_df.iterrows():
        tmdb_id = int(row['id'])
        ml_id = tmdb_to_ml.get(tmdb_id)
        parts = []
        if ml_id and ml_id in user_tags:
            parts.append(user_tags[ml_id])
        if ml_id and ml_id in genome_top:
            parts.append(genome_top[ml_id])
        tag_col.append(' '.join(parts) if parts else '')

    movies_df['ml_tags'] = tag_col
    enriched = sum(1 for t in tag_col if t)
    print(f"    {enriched}/{len(movies_df)} movies got new tag features.")
    return movies_df


def update_ratings(movies_df, tmdb_to_ml):
    """Merge MovieLens aggregate ratings (25M ratings, more reliable)."""
    print("[3/6] Computing aggregate ratings from 25M user ratings...")
    ratings = pd.read_csv(ML_RATINGS, usecols=['movieId', 'rating'])

    our_ml_ids = set()
    for tmdb_id in movies_df['id'].astype(int):
        if tmdb_id in tmdb_to_ml:
            our_ml_ids.add(tmdb_to_ml[tmdb_id])

    ratings = ratings[ratings['movieId'].isin(our_ml_ids)]
    agg = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    agg.columns = ['movieId', 'ml_rating', 'ml_count']

    agg['tmdb_id'] = agg['movieId'].map({ml: tmdb for tmdb, ml in tmdb_to_ml.items()})
    agg = agg.dropna(subset=['tmdb_id'])
    agg['tmdb_id'] = agg['tmdb_id'].astype(int)

    # Build lookup
    ml_ratings = dict(zip(agg['tmdb_id'], agg['ml_rating']))
    ml_counts  = dict(zip(agg['tmdb_id'], agg['ml_count']))

    # Blend: use MovieLens rating when we have enough votes (>50)
    # Scale MovieLens (0.5–5.0) to TMDB scale (0–10)
    updated = 0
    for idx, row in movies_df.iterrows():
        tmdb_id = int(row['id'])
        if tmdb_id in ml_ratings and ml_counts[tmdb_id] >= 50:
            ml_r = ml_ratings[tmdb_id] * 2.0  # 5-point → 10-point
            tmdb_r = float(row['vote_average'])
            tmdb_c = float(row.get('vote_count', 0))
            ml_c   = ml_counts[tmdb_id]
            # Weighted blend by vote count
            total = tmdb_c + ml_c
            blended = (tmdb_r * tmdb_c + ml_r * ml_c) / total if total > 0 else tmdb_r
            movies_df.at[idx, 'vote_average'] = round(blended, 2)
            movies_df.at[idx, 'vote_count'] = int(tmdb_c + ml_c)
            updated += 1

    # Recompute Bayesian weighted_rating
    C = movies_df['vote_average'].mean()
    m = 500
    movies_df['weighted_rating'] = (
        (movies_df['vote_count'] / (movies_df['vote_count'] + m)) * movies_df['vote_average'] +
        (m / (movies_df['vote_count'] + m)) * C
    ).round(3)

    print(f"    {updated} movies got blended ratings.")
    return movies_df


def rebuild_tfidf_and_similarity(movies_df):
    """Rebuild TF-IDF matrix and content similarity with enriched features."""
    print("[4/6] Rebuilding TF-IDF with enriched features (genres+keywords+cast+director+tags+overview)...")

    def build_soup(row):
        parts = []
        # Genres ×3
        genres = str(row.get('genres_display', ''))
        for g in genres.split(','):
            g = g.strip().lower().replace(' ', '')
            if g:
                parts.extend([g] * 3)
        # Keywords ×2
        kw = str(row.get('keywords', ''))
        for k in kw.split(','):
            k = k.strip().lower().replace(' ', '')
            if k:
                parts.extend([k] * 2)
        # Director ×3
        director = str(row.get('director', ''))
        for d in director.split(','):
            d = d.strip().lower().replace(' ', '')
            if d:
                parts.extend([d] * 3)
        # Cast ×1
        cast = str(row.get('cast', ''))
        for c in cast.split():
            c = c.strip().lower()
            if c:
                parts.append(c)
        # Production companies ×1
        companies = str(row.get('production_companies', ''))
        for c in companies.split(','):
            c = c.strip().lower().replace(' ', '')
            if c:
                parts.append(c)
        # Overview ×1
        overview = str(row.get('overview', ''))
        if overview and overview != 'nan':
            parts.append(overview.lower())
        # MovieLens tags ×2 (enrichment from ML25M)
        ml_tags = str(row.get('ml_tags', ''))
        if ml_tags and ml_tags != 'nan':
            parts.extend([ml_tags] * 2)
        return ' '.join(parts)

    movies_df['soup'] = movies_df.apply(build_soup, axis=1)

    vectorizer = TfidfVectorizer(
        max_features=20000,   # increased from 15k to accommodate tags
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(movies_df['soup'])
    print(f"    TF-IDF matrix: {tfidf_matrix.shape}")

    # Save TF-IDF
    with open(TFIDF_PKL, 'wb') as f:
        pickle.dump(tfidf_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(TFIDF_VEC_PKL, 'wb') as f:
        pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"    Saved TF-IDF matrix → {TFIDF_PKL}")

    # Content similarity matrix
    print("    Computing content cosine similarity (this may take a few minutes)...")
    sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    with open(SIM_PKL, 'wb') as f:
        pickle.dump(sim, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"    Saved content similarity → {SIM_PKL}")

    # Drop soup column before saving CSV
    movies_df.drop(columns=['soup'], inplace=True)
    return movies_df


def build_collaborative_similarity(movies_df, tmdb_to_ml, ml_to_tmdb):
    """
    Build item-item collaborative similarity from MovieLens 25M ratings.

    Strategy:
      - Build sparse user-item rating matrix for our movies only
      - Compute item-item cosine similarity in sparse space
      - Store as sparse CSR matrix (memory-efficient)
    """
    print("[5/6] Building collaborative similarity from 25M ratings...")

    # Map our TMDB IDs to MovieLens IDs
    our_tmdb_ids = list(movies_df['id'].astype(int))
    tmdb_to_pos = {int(tmdb_id): pos for pos, tmdb_id in enumerate(our_tmdb_ids)}

    valid_ml_ids = {}
    for tmdb_id in our_tmdb_ids:
        if tmdb_id in tmdb_to_ml:
            valid_ml_ids[tmdb_to_ml[tmdb_id]] = tmdb_to_pos[tmdb_id]

    print(f"    {len(valid_ml_ids)} of {len(our_tmdb_ids)} movies have MovieLens ratings.")

    # Load ratings in chunks (647MB file)
    print("    Loading ratings (chunked, ~25M rows)...")
    n_items = len(our_tmdb_ids)
    rows, cols, data = [], [], []
    user_id_map = {}
    next_user_idx = 0

    for chunk in pd.read_csv(ML_RATINGS, chunksize=1_000_000):
        chunk = chunk[chunk['movieId'].isin(valid_ml_ids)]
        for _, r in chunk.iterrows():
            uid = int(r['userId'])
            if uid not in user_id_map:
                user_id_map[uid] = next_user_idx
                next_user_idx += 1
            rows.append(user_id_map[uid])
            cols.append(valid_ml_ids[int(r['movieId'])])
            data.append(float(r['rating']))

    n_users = next_user_idx
    print(f"    Ratings loaded: {len(data)} ratings from {n_users} users for {len(valid_ml_ids)} items.")

    # Build sparse user-item matrix
    user_item = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )

    # Normalize rows (L2) for cosine similarity
    print("    Computing item-item cosine similarity (sparse)...")
    # Item-item similarity: compute on item×user matrix (transposed)
    # cosine_similarity on sparse matrices is efficient in sklearn
    item_user = user_item.T.tocsr()  # (n_items, n_users)

    # Compute in blocks to manage memory (n_items × n_items can be large)
    BLOCK = 500
    n = n_items
    collab_sim = np.zeros((n, n), dtype=np.float32)
    for start in range(0, n, BLOCK):
        end = min(start + BLOCK, n)
        block = cosine_similarity(item_user[start:end], item_user)
        collab_sim[start:end] = block
        if (start // BLOCK) % 4 == 0:
            print(f"    ... block {start}-{end}/{n}")

    print(f"    Collaborative similarity matrix: {collab_sim.shape}")

    # Save
    with open(COLLAB_SIM_PKL, 'wb') as f:
        pickle.dump(collab_sim, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"    Saved → {COLLAB_SIM_PKL}")

    return collab_sim


def main():
    print("=" * 60)
    print("Maple — MovieLens 25M Integration")
    print("=" * 60)

    # Verify files exist
    for f in [ML_LINKS, ML_RATINGS, ML_TAGS, ML_GENOME_S, ML_GENOME_T, MOVIES_CSV]:
        if not os.path.exists(f):
            print(f"ERROR: Missing file: {f}")
            sys.exit(1)

    # Backup existing CSV
    backup = MOVIES_CSV + '.bak2'
    if not os.path.exists(backup):
        import shutil
        shutil.copy2(MOVIES_CSV, backup)
        print(f"Backed up → {backup}")

    # Load existing movies
    movies_df = pd.read_csv(MOVIES_CSV)
    print(f"Existing movies: {len(movies_df)}")

    # Step 1: ID mapping
    ml_to_tmdb, tmdb_to_ml = load_id_mapping()

    # Step 2: Enrich with tags
    movies_df = enrich_with_tags(movies_df, tmdb_to_ml)

    # Step 3: Blend ratings
    movies_df = update_ratings(movies_df, tmdb_to_ml)

    # Step 4: Rebuild TF-IDF + content similarity
    movies_df = rebuild_tfidf_and_similarity(movies_df)

    # Step 5: Build collaborative similarity
    build_collaborative_similarity(movies_df, tmdb_to_ml, ml_to_tmdb)

    # Step 6: Save updated CSV
    print("[6/6] Saving enriched CSV...")
    movies_df.to_csv(MOVIES_CSV, index=False)
    print(f"    Saved → {MOVIES_CSV}")

    print()
    print("=" * 60)
    print("Integration complete!")
    print(f"  Movies:               {len(movies_df)}")
    print(f"  Tag-enriched:         {sum(movies_df['ml_tags'].fillna('').str.len() > 0)}")
    print(f"  Content sim matrix:   {SIM_PKL}")
    print(f"  Collab sim matrix:    {COLLAB_SIM_PKL}")
    print(f"  TF-IDF matrix:        {TFIDF_PKL}")
    print("=" * 60)


if __name__ == '__main__':
    main()
