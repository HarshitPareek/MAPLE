"""
Preprocesses the TMDB 2023 930k-movies dataset.
Run once before starting the app:
    python -m app.ml.preprocess_movies2

Improvements over v1:
  - Feature weighting: genres×3, keywords×2, director×3, cast×1, companies×1
  - Text cleaning: punctuation removed, lowercased
  - Saves sparse TF-IDF matrix + vectorizer for proper user-preference-vector personalisation
  - Bayesian weighted rating saved alongside raw vote_average
"""
import os
import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CSV = '/Users/harshitpareek/.cache/kagglehub/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies/versions/897/TMDB_movie_dataset_v11.csv'
TMDB_IMG = 'https://image.tmdb.org/t/p/w500'

BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')


def clean(text):
    """Lowercase and strip punctuation."""
    text = str(text or '').lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def tokenise_csv_field(val, limit=None):
    """Turn a comma-separated field into space-separated tokens, no spaces inside tokens."""
    tokens = [re.sub(r'\s+', '', clean(t)) for t in str(val or '').split(',') if t.strip()]
    if limit:
        tokens = tokens[:limit]
    return ' '.join(tokens)


def build_soup(row):
    """
    Weighted feature soup:
      genres   × 3  — strongest signal
      keywords × 2  — strong signal
      director × 3  — strong creative signature
      cast     × 1  — top-5 cast members
      companies× 1  — mild signal
      overview × 1  — semantic context
    """
    genres    = tokenise_csv_field(row.get('genres', ''))
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
    """
    IMDB-style weighted rating:  WR = (v/(v+m)) * R + (m/(v+m)) * C
    v = vote_count, m = min-vote threshold, R = movie rating, C = global mean
    """
    C = df['vote_average'].mean()
    v = df['vote_count']
    R = df['vote_average']
    return (v / (v + m)) * R + (m / (v + m)) * C


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print("Loading CSV…")
    df = pd.read_csv(CSV, low_memory=False)
    print(f"Total rows: {len(df)}")

    # Quality filter — stricter vote_count floor for Bayesian stability
    df['vote_count']   = pd.to_numeric(df['vote_count'],   errors='coerce').fillna(0)
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0)

    df = df[
        (df['vote_count']   >= 300) &
        (df['vote_average'] >= 5.0) &
        (df['status']       == 'Released') &
        (df['adult']        == False) &
        (df['overview'].notna()) &
        (df['overview'].str.len() > 20) &
        (df['poster_path'].notna())
    ].copy()
    print(f"After filter: {len(df)} movies")

    df = df.reset_index(drop=True)
    df['id']           = df['id'].astype(int)
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
    df['poster']       = TMDB_IMG + df['poster_path'].str.strip()
    df['genres_display'] = df['genres'].fillna('')

    # Bayesian weighted rating
    df['weighted_rating'] = bayesian_rating(df).round(3)

    clean_cols = [
        'id', 'title', 'overview', 'genres_display', 'poster',
        'vote_average', 'vote_count', 'weighted_rating', 'release_year',
        'imdb_id', 'keywords', 'production_companies', 'runtime',
    ]
    # Include cast/director if available in this dataset version
    for col in ('cast', 'director'):
        if col in df.columns:
            clean_cols.append(col)

    clean_df = df[clean_cols].copy()

    # Build TF-IDF soup
    print("Building soup + TF-IDF…")
    df['soup'] = df.apply(build_soup, axis=1)

    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=15000,
        ngram_range=(1, 2),   # bigrams capture "science fiction", "based on" etc.
        min_df=2,              # ignore terms appearing in fewer than 2 movies
        sublinear_tf=True,     # dampen very frequent terms
    )
    tfidf_matrix = tfidf.fit_transform(df['soup'])

    print(f"Computing cosine similarity for {len(clean_df)} movies…")
    sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Save everything
    csv_path        = os.path.join(PROCESSED_DIR, 'processed_movies2.csv')
    sim_path        = os.path.join(PROCESSED_DIR, 'similarity_matrix2.pkl')
    tfidf_mat_path  = os.path.join(PROCESSED_DIR, 'tfidf_matrix2.pkl')
    tfidf_vec_path  = os.path.join(PROCESSED_DIR, 'tfidf_vectorizer2.pkl')

    clean_df.to_csv(csv_path, index=False)
    print(f"Saved {len(clean_df)} movies → {csv_path}")

    with open(sim_path, 'wb') as f:
        pickle.dump(sim, f)
    print(f"Saved similarity matrix → {sim_path}")

    with open(tfidf_mat_path, 'wb') as f:
        pickle.dump(tfidf_matrix, f)      # scipy sparse — small file
    print(f"Saved TF-IDF matrix → {tfidf_mat_path}")

    with open(tfidf_vec_path, 'wb') as f:
        pickle.dump(tfidf, f)
    print(f"Saved TF-IDF vectorizer → {tfidf_vec_path}")


if __name__ == '__main__':
    main()
