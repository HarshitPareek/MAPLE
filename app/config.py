import os
from datetime import timedelta

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_PROC    = os.path.join(BASE_DIR, 'data', 'processed')

class Config:
    SECRET_KEY                  = os.environ.get('SECRET_KEY',     'maple-secret-key-2024')
    JWT_SECRET_KEY              = os.environ.get('JWT_SECRET_KEY', 'maple-jwt-secret-2024')
    JWT_ACCESS_TOKEN_EXPIRES    = timedelta(days=7)
    SQLALCHEMY_DATABASE_URI     = f"sqlite:///{os.path.join(BASE_DIR, 'database', 'app.db')}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Movies (TMDB 2023 dataset + MovieLens 25M enrichment)
    PROCESSED_MOVIES_PATH       = os.path.join(_PROC, 'processed_movies2.csv')
    SIMILARITY_MATRIX_PATH      = os.path.join(_PROC, 'similarity_matrix2.pkl')
    TFIDF_MATRIX_MOVIES_PATH    = os.path.join(_PROC, 'tfidf_matrix2.pkl')
    COLLAB_SIM_MOVIES_PATH      = os.path.join(_PROC, 'collab_similarity_movies.pkl')

    # TV shows
    PROCESSED_TV_PATH           = os.path.join(_PROC, 'processed_tv.csv')
    SIMILARITY_MATRIX_TV_PATH   = os.path.join(_PROC, 'similarity_matrix_tv.pkl')
    TFIDF_MATRIX_TV_PATH        = os.path.join(_PROC, 'tfidf_matrix_tv.pkl')
