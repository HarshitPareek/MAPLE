"""
Run this script ONCE before starting the app:
    python -m app.ml.preprocess

It merges movie metadata, keywords, and credits CSVs,
builds a TF-IDF soup, computes cosine similarity, and saves pickles.

Improvements:
  - Feature weighting: genres×3, keywords×2, director×3, cast×1
  - Text cleaning: punctuation removed, lowercased
  - Bayesian weighted rating saved
  - Saves sparse TF-IDF matrix + vectorizer for proper personalisation
"""
import os
import re
import ast
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

_raw_candidates = [os.path.join(BASE_DIR, 'data', 'raw'), os.path.join(BASE_DIR, 'database')]
RAW_DIR = next((d for d in _raw_candidates if os.path.exists(os.path.join(d, 'movies_metadata.csv'))), _raw_candidates[0])
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')


def safe_parse(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return []


def clean(text):
    """Lowercase and strip punctuation."""
    text = str(text or '').lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def extract_names(obj_list, key='name', limit=None):
    if not isinstance(obj_list, list):
        return []
    items = [re.sub(r'\s+', '', clean(str(o.get(key, '')))) for o in obj_list if isinstance(o, dict)]
    return items[:limit] if limit else items


def get_director(crew_list):
    if not isinstance(crew_list, list):
        return ''
    for member in crew_list:
        if isinstance(member, dict) and member.get('job') == 'Director':
            return re.sub(r'\s+', '', clean(str(member.get('name', ''))))
    return ''


def build_soup(row):
    """
    Weighted feature soup:
      genres   × 3  — strongest signal
      keywords × 2  — strong signal
      director × 3  — strong creative signature
      cast     × 1  — top-5 cast members
      overview × 1  — semantic context
    """
    genres   = ' '.join(extract_names(row['genres']))
    keywords = ' '.join(extract_names(row['keywords'], limit=15))
    cast     = ' '.join(extract_names(row['cast'], limit=5))
    director = re.sub(r'\s+', '', clean(str(row['director'])))
    overview = clean(str(row['overview'])) if pd.notna(row['overview']) else ''

    return (
        f"{genres} {genres} {genres} "
        f"{keywords} {keywords} "
        f"{director} {director} {director} "
        f"{cast} "
        f"{overview}"
    )


def bayesian_rating(df, m=50):
    """IMDB-style weighted rating."""
    C = df['vote_average'].mean()
    v = df['vote_count']
    R = df['vote_average']
    return (v / (v + m)) * R + (m / (v + m)) * C


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print("Loading CSVs...")

    meta     = pd.read_csv(os.path.join(RAW_DIR, 'movies_metadata.csv'), low_memory=False)
    keywords = pd.read_csv(os.path.join(RAW_DIR, 'keywords.csv'))
    credits  = pd.read_csv(os.path.join(RAW_DIR, 'credits.csv'))

    links_path = os.path.join(BASE_DIR, 'database', 'links.csv')
    if not os.path.exists(links_path):
        links_path = os.path.join(RAW_DIR, 'links.csv')
    links = pd.read_csv(links_path)[['tmdbId', 'imdbId']].dropna()
    links = links.assign(tmdbId=links['tmdbId'].astype(int), imdbId=links['imdbId'].astype(int))

    meta     = meta[meta['id'].apply(lambda x: str(x).isdigit())].copy()
    meta     = meta.assign(id=meta['id'].astype(int))
    keywords = keywords.assign(id=keywords['id'].astype(int))
    credits  = credits.assign(id=credits['id'].astype(int))

    meta['vote_count']   = meta['vote_count'].apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
    meta['vote_average'] = meta['vote_average'].apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)

    meta = meta[meta['vote_count'] >= 50]
    meta = meta[meta['overview'].notna() & (meta['overview'].str.len() > 20)]

    df = meta.merge(keywords, on='id').merge(credits, on='id').copy()

    print("Parsing columns...")
    df = df.assign(
        genres=df['genres'].apply(safe_parse),
        keywords=df['keywords'].apply(safe_parse),
        cast=df['cast'].apply(safe_parse),
        crew=df['crew'].apply(safe_parse),
    )
    df = df.assign(director=df['crew'].apply(get_director))

    df = df.assign(
        poster_path=df['poster_path'].fillna(''),
        release_year=pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int),
        vote_average=pd.to_numeric(df['vote_average'], errors='coerce').fillna(0),
        vote_count=pd.to_numeric(df['vote_count'], errors='coerce').fillna(0),
    )
    df = df.assign(genres_display=df['genres'].apply(lambda g: ', '.join(extract_names(g))))

    df = df.merge(links.rename(columns={'tmdbId': 'id'}), on='id', how='left')
    df = df.assign(imdb_id=df['imdbId'].fillna(0).astype(int))

    # Bayesian weighted rating
    df['weighted_rating'] = bayesian_rating(df).round(3)

    clean_df = df[[
        'id', 'title', 'overview', 'genres_display', 'poster_path', 'imdb_id',
        'vote_average', 'vote_count', 'weighted_rating', 'release_year',
        'director', 'genres', 'keywords', 'cast',
    ]].copy().reset_index(drop=True)

    # Build TF-IDF soup
    print("Building recommendation soup...")
    df_indexed = df.reset_index(drop=True)
    df_indexed['soup'] = df_indexed.apply(build_soup, axis=1)

    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=12000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    tfidf_matrix = tfidf.fit_transform(df_indexed['soup'])

    print(f"Computing cosine similarity for {len(clean_df)} movies...")
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    clean_path      = os.path.join(PROCESSED_DIR, 'processed_movies.csv')
    sim_path        = os.path.join(PROCESSED_DIR, 'similarity_matrix.pkl')
    tfidf_mat_path  = os.path.join(PROCESSED_DIR, 'tfidf_matrix.pkl')
    tfidf_vec_path  = os.path.join(PROCESSED_DIR, 'tfidf_vectorizer.pkl')

    clean_df.to_csv(clean_path, index=False)
    print(f"Saved {len(clean_df)} movies → {clean_path}")

    with open(sim_path, 'wb') as f:
        pickle.dump(sim_matrix, f)
    print(f"Saved similarity matrix → {sim_path}")

    with open(tfidf_mat_path, 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    print(f"Saved TF-IDF matrix → {tfidf_mat_path}")

    with open(tfidf_vec_path, 'wb') as f:
        pickle.dump(tfidf, f)
    print(f"Saved TF-IDF vectorizer → {tfidf_vec_path}")


if __name__ == '__main__':
    main()
