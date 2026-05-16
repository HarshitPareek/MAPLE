"""
Run this script ONCE to process the TV shows dataset:
    python -m app.ml.preprocess_tv

Improvements over v1:
  - Feature weighting: genres×3, cast×1, created_by×3, keywords×2, overview×1
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

TV_CSV = '/Users/harshitpareek/.cache/kagglehub/datasets/asaniczka/full-tmdb-tv-shows-dataset-2023-150k-shows/versions/22/TMDB_tv_dataset_v3.csv'
BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
TMDB_IMG      = 'https://image.tmdb.org/t/p/w500'


def clean(text):
    """Lowercase and strip punctuation."""
    text = str(text or '').lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def tokenise_csv_field(val, limit=None):
    """Turn a comma-separated field into no-space tokens."""
    tokens = [re.sub(r'\s+', '', clean(t)) for t in str(val or '').split(',') if t.strip()]
    if limit:
        tokens = tokens[:limit]
    return ' '.join(tokens)


def build_soup(row):
    """
    Weighted feature soup for TV:
      genres     × 3  — strongest signal
      created_by × 3  — showrunner/creator is the TV equivalent of director
      cast       × 1  — top-5 cast
      networks   × 1  — mild distribution signal
      keywords   × 2  — if available
      overview   × 1  — semantic context
    """
    genres     = tokenise_csv_field(row.get('genres', ''))
    created_by = tokenise_csv_field(row.get('created_by', ''))
    cast       = tokenise_csv_field(row.get('cast', ''), limit=5)
    networks   = tokenise_csv_field(row.get('networks', ''), limit=2)
    keywords   = tokenise_csv_field(row.get('keywords', ''), limit=10)
    overview   = clean(row.get('overview', ''))

    return (
        f"{genres} {genres} {genres} "
        f"{created_by} {created_by} {created_by} "
        f"{cast} "
        f"{networks} "
        f"{keywords} {keywords} "
        f"{overview}"
    )


def bayesian_rating(df, m=200):
    """IMDB-style weighted rating."""
    C = df['vote_average'].mean()
    v = df['vote_count']
    R = df['vote_average']
    return (v / (v + m)) * R + (m / (v + m)) * C


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if not os.path.exists(TV_CSV):
        print(f"[TV] Dataset not found at {TV_CSV}")
        print("Run: python3 -c \"import kagglehub; kagglehub.dataset_download('asaniczka/full-tmdb-tv-shows-dataset-2023-150k-shows')\"")
        return

    print("[TV] Loading TV dataset...")
    df = pd.read_csv(TV_CSV)
    print(f"[TV] Total rows: {len(df)}")

    df['vote_count']   = pd.to_numeric(df['vote_count'],   errors='coerce').fillna(0)
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0)

    # Filter: min votes, has overview, not adult
    df = df[
        (df['vote_count']   >= 50) &
        (df['vote_average'] >= 5.0) &
        df['overview'].notna() &
        (df['overview'].str.len() > 20) &
        (df['adult'] == False)
    ].copy()
    print(f"[TV] After filtering: {len(df)}")

    df = df.reset_index(drop=True)

    # Bayesian weighted rating
    df['weighted_rating'] = bayesian_rating(df).round(3)

    df = df.assign(
        release_year=pd.to_datetime(df['first_air_date'], errors='coerce').dt.year.fillna(0).astype(int),
        poster_path=df['poster_path'].fillna(''),
        number_of_seasons=pd.to_numeric(df['number_of_seasons'], errors='coerce').fillna(0).astype(int),
        number_of_episodes=pd.to_numeric(df['number_of_episodes'], errors='coerce').fillna(0).astype(int),
        genres_display=df['genres'].fillna(''),
        networks=df['networks'].fillna(''),
        created_by=df['created_by'].fillna(''),
        status=df['status'].fillna(''),
    )

    df = df.assign(
        poster=df['poster_path'].apply(
            lambda p: (TMDB_IMG + p) if p and str(p).startswith('/') else ''
        )
    )

    # Only keep rows that have a poster (improves UI quality)
    df = df[df['poster'] != ''].copy()
    print(f"[TV] After poster filter: {len(df)}")

    clean_cols = [
        'id', 'name', 'overview', 'genres_display', 'poster',
        'vote_average', 'vote_count', 'weighted_rating', 'release_year',
        'number_of_seasons', 'number_of_episodes', 'networks', 'created_by', 'status',
    ]
    for col in ('cast', 'keywords'):
        if col in df.columns:
            clean_cols.append(col)

    clean_df = df[clean_cols].copy()
    clean_df = clean_df.rename(columns={'name': 'title'})

    # Build TF-IDF soup
    print("[TV] Building TF-IDF soup...")
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

    print(f"[TV] Computing cosine similarity for {len(clean_df)} shows...")
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Save
    tv_csv_path     = os.path.join(PROCESSED_DIR, 'processed_tv.csv')
    sim_path        = os.path.join(PROCESSED_DIR, 'similarity_matrix_tv.pkl')
    tfidf_mat_path  = os.path.join(PROCESSED_DIR, 'tfidf_matrix_tv.pkl')
    tfidf_vec_path  = os.path.join(PROCESSED_DIR, 'tfidf_vectorizer_tv.pkl')

    clean_df.to_csv(tv_csv_path, index=False)
    print(f"[TV] Saved {len(clean_df)} shows → {tv_csv_path}")

    with open(sim_path, 'wb') as f:
        pickle.dump(sim_matrix, f)
    print(f"[TV] Saved similarity matrix → {sim_path}")

    with open(tfidf_mat_path, 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    print(f"[TV] Saved TF-IDF matrix → {tfidf_mat_path}")

    with open(tfidf_vec_path, 'wb') as f:
        pickle.dump(tfidf, f)
    print(f"[TV] Saved TF-IDF vectorizer → {tfidf_vec_path}")


if __name__ == '__main__':
    main()
