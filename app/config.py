import os
import secrets
from datetime import timedelta

from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_PROC    = os.path.join(BASE_DIR, 'data', 'processed')
_ENV_FILE = os.path.join(BASE_DIR, '.env')

load_dotenv(_ENV_FILE)


def _secret(name):
    """Return the signing secret for *name*, generating one if absent.

    Order of preference:
      1. the environment (or .env, loaded above) — how deployments supply it
      2. a freshly generated 32-byte secret, appended to a gitignored .env

    Generating on first run keeps the secret out of the repo without breaking
    a plain ``python run.py`` checkout. Persisting it matters: a per-process
    random key would invalidate every JWT on restart.
    """
    existing = os.environ.get(name)
    if existing:
        return existing

    value = secrets.token_urlsafe(32)
    try:
        with open(_ENV_FILE, 'a', encoding='utf-8') as fh:
            fh.write(f'{name}={value}\n')
    except OSError:
        # Read-only checkout: fall back to a process-lifetime secret. Sessions
        # and tokens then reset on restart, which beats refusing to boot.
        pass
    os.environ[name] = value
    return value


class Config:
    SECRET_KEY                  = _secret('SECRET_KEY')
    JWT_SECRET_KEY              = _secret('JWT_SECRET_KEY')
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
