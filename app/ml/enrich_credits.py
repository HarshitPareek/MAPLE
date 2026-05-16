"""
Maple — Enrich movies + TV with cast/director from credits.csv

Parses the TMDB credits.csv (JSON cast/crew), fills missing cast/director
fields in processed_movies2.csv, and adds cast column to processed_tv.csv.

Run:  python -m app.ml.enrich_credits
"""

import os
import ast
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CREDITS_CSV = os.path.join(BASE_DIR, 'database', 'credits.csv')
MOVIES_CSV  = os.path.join(BASE_DIR, 'data', 'processed', 'processed_movies2.csv')
TV_CSV      = os.path.join(BASE_DIR, 'data', 'processed', 'processed_tv.csv')
TMDB_IMG    = 'https://image.tmdb.org/t/p/w185'


def parse_credits():
    """Parse credits.csv → {tmdb_id: {cast: [...], director: str}}"""
    print("[1/3] Parsing credits.csv...")
    credits = pd.read_csv(CREDITS_CSV)
    result = {}

    for _, row in credits.iterrows():
        tmdb_id = int(row['id'])
        # Parse cast
        cast_list = []
        try:
            raw = ast.literal_eval(str(row['cast']))
            for c in raw[:10]:  # top 10 cast
                entry = {
                    'name': c.get('name', ''),
                    'character': c.get('character', ''),
                    'profile': (TMDB_IMG + c['profile_path']) if c.get('profile_path') else '',
                }
                cast_list.append(entry)
        except Exception:
            pass

        # Parse director from crew
        director = ''
        try:
            crew = ast.literal_eval(str(row['crew']))
            directors = [c['name'] for c in crew if c.get('job') == 'Director']
            director = ', '.join(directors[:3])
        except Exception:
            pass

        result[tmdb_id] = {
            'cast_json': str(cast_list) if cast_list else '',
            'cast_names': ', '.join(c['name'] for c in cast_list) if cast_list else '',
            'director': director,
        }

    print(f"    Parsed {len(result)} entries.")
    return result


def enrich_movies(credits_map):
    """Fill missing cast/director in movies CSV and add cast_json column."""
    print("[2/3] Enriching movies...")
    df = pd.read_csv(MOVIES_CSV)

    if 'cast_json' not in df.columns:
        df['cast_json'] = ''

    updated_cast = 0
    updated_dir = 0

    for idx, row in df.iterrows():
        tmdb_id = int(row['id'])
        if tmdb_id not in credits_map:
            continue

        cdata = credits_map[tmdb_id]

        # Fill cast names (space-separated for TF-IDF soup compatibility)
        current_cast = str(row.get('cast', '') or '').strip()
        if (not current_cast or current_cast == 'nan') and cdata['cast_names']:
            df.at[idx, 'cast'] = ' '.join(cdata['cast_names'].replace(',', '').split())
            updated_cast += 1

        # Fill director
        current_dir = str(row.get('director', '') or '').strip()
        if (not current_dir or current_dir == 'nan') and cdata['director']:
            df.at[idx, 'director'] = cdata['director']
            updated_dir += 1

        # Always update cast_json (structured data for UI)
        if cdata['cast_json']:
            df.at[idx, 'cast_json'] = cdata['cast_json']

    df.to_csv(MOVIES_CSV, index=False)
    has_cast = df['cast'].fillna('').str.strip().str.len() > 0
    has_dir = df['director'].fillna('').str.strip().str.len() > 0
    print(f"    Cast: filled {updated_cast} new → {has_cast.sum()}/{len(df)} total")
    print(f"    Director: filled {updated_dir} new → {has_dir.sum()}/{len(df)} total")
    print(f"    Cast JSON: {(df['cast_json'].fillna('').str.len() > 0).sum()}/{len(df)} have structured data")


def enrich_tv(credits_map):
    """Add cast and cast_json columns to TV CSV."""
    print("[3/3] Enriching TV shows...")
    df = pd.read_csv(TV_CSV)

    if 'cast' not in df.columns:
        df['cast'] = ''
    if 'cast_json' not in df.columns:
        df['cast_json'] = ''
    if 'director' not in df.columns:
        df['director'] = ''

    updated = 0
    for idx, row in df.iterrows():
        tmdb_id = int(row['id'])
        if tmdb_id not in credits_map:
            continue

        cdata = credits_map[tmdb_id]

        if cdata['cast_names']:
            df.at[idx, 'cast'] = ' '.join(cdata['cast_names'].replace(',', '').split())
            updated += 1
        if cdata['cast_json']:
            df.at[idx, 'cast_json'] = cdata['cast_json']
        if cdata['director'] and not str(row.get('created_by', '') or '').strip():
            df.at[idx, 'director'] = cdata['director']

    df.to_csv(TV_CSV, index=False)
    has_cast = df['cast'].fillna('').str.strip().str.len() > 0
    print(f"    TV cast enriched: {updated} new → {has_cast.sum()}/{len(df)} total")


def main():
    print("=" * 50)
    print("Maple — Cast/Director Enrichment")
    print("=" * 50)
    credits_map = parse_credits()
    enrich_movies(credits_map)
    enrich_tv(credits_map)
    print("\nDone!")


if __name__ == '__main__':
    main()
