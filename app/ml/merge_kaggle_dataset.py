"""
Merge the Kaggle utkarshx27/movies-dataset into the existing processed_movies2.csv,
then rebuild the TF-IDF matrix and similarity matrix.

Run once:
    python -m app.ml.merge_kaggle_dataset

What this does:
  1. Loads the existing processed_movies2.csv (8,758 movies)
  2. Loads the new Kaggle dataset (4,803 movies)
  3. For overlapping IDs:  enriches existing rows with cast & director
  4. For new-only IDs:     converts & appends (genres, companies, poster, etc.)
  5. Rebuilds TF-IDF + similarity matrix over the merged set
"""

import os
import re
import ast
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ──
BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

EXISTING_CSV  = os.path.join(PROCESSED_DIR, 'processed_movies2.csv')
KAGGLE_CSV    = os.path.expanduser(
    '~/.cache/kagglehub/datasets/utkarshx27/movies-dataset/versions/1/movie_dataset.csv'
)
TMDB_SOURCE   = os.path.expanduser(
    '~/.cache/kagglehub/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies/versions/897/TMDB_movie_dataset_v11.csv'
)
TMDB_IMG      = 'https://image.tmdb.org/t/p/w500'


# ── Helpers ──

def clean(text):
    text = str(text or '').lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def tokenise_csv_field(val, limit=None):
    tokens = [re.sub(r'\s+', '', clean(t)) for t in str(val or '').split(',') if t.strip()]
    if limit:
        tokens = tokens[:limit]
    return ' '.join(tokens)


def build_soup(row):
    genres    = tokenise_csv_field(row.get('genres_display', ''))
    keywords  = tokenise_csv_field(row.get('keywords', ''), limit=15)
    director  = tokenise_csv_field(row.get('director', ''))
    cast      = tokenise_csv_field(row.get('cast', ''), limit=5)
    companies = tokenise_csv_field(row.get('production_companies', ''), limit=3)
    overview  = clean(row.get('overview', ''))

    return (
        f"{genres} {genres} {genres} "
        f"{keywords} {keywords} "
        f"{director} {director} {director} "
        f"{cast} "
        f"{companies} "
        f"{overview}"
    )


def bayesian_rating(df, m=500):
    C = df['vote_average'].mean()
    v = df['vote_count'].fillna(0)
    R = df['vote_average']
    return (v / (v + m)) * R + (m / (v + m)) * C


def parse_json_names(raw):
    """Extract 'name' values from a JSON-like list string, e.g. [{"name":"Foo",...}]."""
    if pd.isna(raw) or not str(raw).strip():
        return ''
    try:
        items = ast.literal_eval(str(raw))
        if isinstance(items, list):
            return ', '.join(item['name'] for item in items if isinstance(item, dict) and 'name' in item)
    except (ValueError, SyntaxError):
        pass
    return ''


def space_to_comma_genres(raw):
    """
    The Kaggle dataset has space-separated multi-word genre names like
    'Action Adventure Science Fiction'.  We need to map them to the
    canonical TMDB comma-separated format.
    """
    if pd.isna(raw) or not str(raw).strip():
        return ''
    text = str(raw).strip()

    # Known multi-word genres (order matters — longest first)
    MULTI_WORD = [
        'Science Fiction', 'TV Movie', 'War & Politics',
        'Sci-Fi & Fantasy', 'Action & Adventure',
        'Kids & Family', 'War & Politics',
    ]
    # Replace multi-word genres with placeholder tokens
    placeholders = {}
    for i, mw in enumerate(MULTI_WORD):
        token = f'__MW{i}__'
        if mw in text:
            text = text.replace(mw, token)
            placeholders[token] = mw

    # Now remaining words are single-word genres
    parts = text.split()

    # Restore multi-word genres
    result = []
    for p in parts:
        if p in placeholders:
            result.append(placeholders[p])
        else:
            result.append(p)

    return ', '.join(result)


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ── Load existing ──
    print("Loading existing processed_movies2.csv…")
    existing = pd.read_csv(EXISTING_CSV)
    existing['id'] = existing['id'].astype(int)
    print(f"  Existing: {len(existing)} movies")

    # ── Load Kaggle dataset ──
    print("Loading Kaggle dataset…")
    kaggle = pd.read_csv(KAGGLE_CSV)
    kaggle['id'] = kaggle['id'].astype(int)
    print(f"  Kaggle:   {len(kaggle)} movies")

    # ── Load TMDB source for poster paths ──
    poster_map = {}
    if os.path.exists(TMDB_SOURCE):
        print("Loading TMDB source for poster paths…")
        tmdb = pd.read_csv(TMDB_SOURCE, usecols=['id', 'poster_path'], low_memory=False)
        tmdb = tmdb.dropna(subset=['poster_path'])
        tmdb['id'] = pd.to_numeric(tmdb['id'], errors='coerce')
        tmdb = tmdb.dropna(subset=['id'])
        tmdb['id'] = tmdb['id'].astype(int)
        poster_map = dict(zip(tmdb['id'], tmdb['poster_path']))
        print(f"  Poster paths available for {len(poster_map)} TMDB IDs")
    else:
        print("  TMDB source not found — new movies won't have poster images")

    # ── Prepare Kaggle data ──
    # Parse production companies from JSON
    kaggle['production_companies_clean'] = kaggle['production_companies'].apply(parse_json_names)
    # Convert genres from space-separated to comma-separated
    kaggle['genres_clean'] = kaggle['genres'].apply(space_to_comma_genres)
    # Extract year
    kaggle['release_year'] = pd.to_datetime(kaggle['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
    # Clean cast and director
    kaggle['cast_clean'] = kaggle['cast'].fillna('')
    kaggle['director_clean'] = kaggle['director'].fillna('')

    # ── Step 1: Enrich overlapping IDs with cast & director ──
    existing_ids = set(existing['id'])
    overlap_ids  = existing_ids & set(kaggle['id'])
    print(f"\nOverlapping IDs: {len(overlap_ids)}")

    if 'cast' not in existing.columns:
        existing['cast'] = ''
    if 'director' not in existing.columns:
        existing['director'] = ''

    kaggle_lookup = kaggle.set_index('id')
    enriched = 0
    for idx, row in existing.iterrows():
        mid = row['id']
        if mid in overlap_ids and mid in kaggle_lookup.index:
            k_row = kaggle_lookup.loc[mid]
            if isinstance(k_row, pd.DataFrame):
                k_row = k_row.iloc[0]
            # Enrich cast & director if missing
            if not str(row.get('cast', '')).strip() and str(k_row.get('cast_clean', '')).strip():
                existing.at[idx, 'cast'] = k_row['cast_clean']
                enriched += 1
            if not str(row.get('director', '')).strip() and str(k_row.get('director_clean', '')).strip():
                existing.at[idx, 'director'] = k_row['director_clean']
                enriched += 1
            # Also enrich keywords if existing is empty
            if not str(row.get('keywords', '')).strip() and str(k_row.get('keywords', '')).strip():
                existing.at[idx, 'keywords'] = str(k_row['keywords'])

    print(f"  Enriched {enriched} fields on existing movies (cast/director)")

    # ── Step 2: Add new-only movies ──
    new_only = kaggle[~kaggle['id'].isin(existing_ids)].copy()
    # Quality filter — skip movies with no overview or very low vote count
    new_only = new_only[
        (new_only['overview'].notna()) &
        (new_only['overview'].str.len() > 20) &
        (new_only['vote_count'] >= 50) &
        (new_only['vote_average'] >= 3.0)
    ].copy()
    print(f"  New movies after quality filter: {len(new_only)}")

    # Build new rows in existing schema
    new_rows = []
    for _, k in new_only.iterrows():
        mid = int(k['id'])
        # Poster: try TMDB source, else empty
        poster = ''
        if mid in poster_map:
            pp = str(poster_map[mid]).strip()
            if pp:
                poster = TMDB_IMG + pp

        new_rows.append({
            'id':                    mid,
            'title':                 str(k['title']),
            'overview':              str(k['overview']),
            'genres_display':        k['genres_clean'],
            'poster':                poster,
            'vote_average':          float(k['vote_average']),
            'vote_count':            float(k.get('vote_count', 0)),
            'release_year':          int(k['release_year']),
            'imdb_id':               '',
            'keywords':              str(k.get('keywords', '')),
            'production_companies':  k['production_companies_clean'],
            'runtime':               float(k['runtime']) if pd.notna(k.get('runtime')) else 0,
            'cast':                  k['cast_clean'],
            'director':              k['director_clean'],
        })

    new_df = pd.DataFrame(new_rows)
    print(f"  Adding {len(new_df)} new movies")

    # ── Step 3: Merge ──
    # Ensure existing has vote_count column (may be missing from v1 processing)
    if 'vote_count' not in existing.columns:
        existing['vote_count'] = 0

    merged = pd.concat([existing, new_df], ignore_index=True)

    # Drop any duplicates by ID (safety)
    merged = merged.drop_duplicates(subset='id', keep='first').reset_index(drop=True)
    print(f"\nMerged total: {len(merged)} movies")

    # ── Step 4: Recompute Bayesian weighted rating ──
    merged['vote_count']   = pd.to_numeric(merged['vote_count'], errors='coerce').fillna(0)
    merged['vote_average'] = pd.to_numeric(merged['vote_average'], errors='coerce').fillna(0)
    merged['weighted_rating'] = bayesian_rating(merged).round(3)

    # ── Step 5: Rebuild TF-IDF + similarity matrix ──
    print("Building soup + TF-IDF…")
    merged['cast']     = merged['cast'].fillna('')
    merged['director'] = merged['director'].fillna('')
    merged['keywords'] = merged['keywords'].fillna('')
    merged['production_companies'] = merged['production_companies'].fillna('')

    soups = merged.apply(build_soup, axis=1)

    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=15000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    tfidf_matrix = tfidf.fit_transform(soups)
    print(f"TF-IDF matrix: {tfidf_matrix.shape}")

    print(f"Computing cosine similarity for {len(merged)} movies…")
    sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print(f"Similarity matrix: {sim.shape}")

    # ── Step 6: Save ──
    # Clean output columns
    output_cols = [
        'id', 'title', 'overview', 'genres_display', 'poster',
        'vote_average', 'vote_count', 'weighted_rating', 'release_year',
        'imdb_id', 'keywords', 'production_companies', 'runtime',
        'cast', 'director',
    ]
    out_df = merged[[c for c in output_cols if c in merged.columns]]

    csv_path       = os.path.join(PROCESSED_DIR, 'processed_movies2.csv')
    sim_path       = os.path.join(PROCESSED_DIR, 'similarity_matrix2.pkl')
    tfidf_mat_path = os.path.join(PROCESSED_DIR, 'tfidf_matrix2.pkl')
    tfidf_vec_path = os.path.join(PROCESSED_DIR, 'tfidf_vectorizer2.pkl')

    # Backup existing
    backup_csv = csv_path + '.bak'
    if os.path.exists(csv_path) and not os.path.exists(backup_csv):
        import shutil
        shutil.copy2(csv_path, backup_csv)
        print(f"  Backed up existing CSV → {backup_csv}")

    out_df.to_csv(csv_path, index=False)
    print(f"Saved {len(out_df)} movies → {csv_path}")

    with open(sim_path, 'wb') as f:
        pickle.dump(sim, f)
    print(f"Saved similarity matrix → {sim_path}")

    with open(tfidf_mat_path, 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    print(f"Saved TF-IDF matrix → {tfidf_mat_path}")

    with open(tfidf_vec_path, 'wb') as f:
        pickle.dump(tfidf, f)
    print(f"Saved TF-IDF vectorizer → {tfidf_vec_path}")

    print("\nDone! Restart the app to load the updated data.")


if __name__ == '__main__':
    main()
