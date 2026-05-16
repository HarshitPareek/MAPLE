"""
Maple recommendation routes — v3

Interest profile builder:
  - 3-tier time decay:  last 7 days → ×1.5,  last 30 days → ×1.0,  older → ×0.5
  - Multi-signal base weights: liked=3, favorite=2, watchlist=1.5, watched=1, view=0.4
  - Exponential recency decay within each tier for fine-grained ordering
  - Implicit interactions (detail views) from last 90 days
"""

from flask import Blueprint, jsonify, request, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import db
from app.models.user_movie import UserMovie
from app.models.user_interaction import UserInteraction
from datetime import datetime, timedelta
import math

rec_bp = Blueprint('rec', __name__)


# ---------------------------------------------------------------------------
# Time-based weight tiers
# ---------------------------------------------------------------------------

# 3-tier multipliers (spec requirement)
#   last 7 days  → 1.5
#   last 30 days → 1.0
#   older        → 0.5
# On top of that, within each tier, apply exponential decay with a 30-day
# half-life so that "3 days ago" beats "6 days ago" even though both are
# in the same tier.

_TIER_CUTOFFS = [
    (7,   1.5),   # days, multiplier
    (30,  1.0),
    (None, 0.5),  # everything older
]
_HALF_LIFE = 30   # days
_LAMBDA    = math.log(2) / _HALF_LIFE

BASE_WEIGHT = {
    'liked':          3.0,
    'favorite':       2.0,
    'watchlist':      1.5,
    'watched':        1.0,
    'not_interested': -4.0,   # strong negative signal
}


def _time_weight(days_old):
    """
    Combined weight:  tier_multiplier × exp(-λ × days)
    """
    for cutoff, mult in _TIER_CUTOFFS:
        if cutoff is None or days_old <= cutoff:
            return mult * math.exp(-_LAMBDA * days_old)
    return 0.5 * math.exp(-_LAMBDA * days_old)


# ---------------------------------------------------------------------------
# Interest profile builder
# ---------------------------------------------------------------------------

def _build_interest_profile(user_id):
    """
    Build weighted seed lists for the recommendation engine.

    Returns:
        movie_ids, tv_ids          — collapsed seed ID lists (positive signals only)
        movie_weights, tv_weights  — parallel float weight arrays
        all_movie_ids, all_tv_ids  — all IDs across every list (for exclusion)
        ni_movie_ids, ni_tv_ids    — "not interested" IDs (hard exclusion + negative signal)
    """
    now = datetime.utcnow()

    # --- Explicit list items ---
    user_movies = UserMovie.query.filter_by(user_id=user_id).all()

    movie_seeds   = []   # [(id, weight), ...]
    tv_seeds      = []
    all_movie_ids = set()
    all_tv_ids    = set()
    ni_movie_ids  = set()
    ni_tv_ids     = set()

    for um in user_movies:
        bucket = all_tv_ids if um.content_type == 'tv' else all_movie_ids
        bucket.add(um.movie_id)

        # Track not-interested separately for hard exclusion
        if um.list_type == 'not_interested':
            ni_bucket = ni_tv_ids if um.content_type == 'tv' else ni_movie_ids
            ni_bucket.add(um.movie_id)

        base = BASE_WEIGHT.get(um.list_type, 0.5)
        days = max((now - um.added_at).total_seconds() / 86400, 0) if um.added_at else 0
        w    = base * _time_weight(days)

        target = tv_seeds if um.content_type == 'tv' else movie_seeds
        target.append((um.movie_id, w))

    # --- Implicit interactions (last 90 days only) ---
    cutoff = now - timedelta(days=90)
    interactions = UserInteraction.query.filter(
        UserInteraction.user_id == user_id,
        UserInteraction.timestamp >= cutoff,
    ).all()

    for ix in interactions:
        base = 0.4 if ix.action == 'view' else 0.2
        days = max((now - ix.timestamp).total_seconds() / 86400, 0)
        w    = base * _time_weight(days)
        target = tv_seeds if ix.content_type == 'tv' else movie_seeds
        target.append((ix.movie_id, w))

    # Collapse: group by ID, sum weights
    # Separate positive seeds (for user vector) from not-interested (only exclude)
    def collapse(seeds, ni_ids):
        by_id = {}
        for mid, w in seeds:
            by_id[mid] = by_id.get(mid, 0) + w
        # Remove not-interested from positive seeds — they're only for exclusion
        for nid in ni_ids:
            by_id.pop(nid, None)
        ids     = list(by_id.keys())
        weights = [by_id[i] for i in ids]
        return ids, weights

    m_ids, m_w = collapse(movie_seeds, ni_movie_ids)
    t_ids, t_w = collapse(tv_seeds, ni_tv_ids)

    return (m_ids, t_ids, m_w, t_w,
            list(all_movie_ids), list(all_tv_ids),
            list(ni_movie_ids), list(ni_tv_ids))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@rec_bp.route('/similar/<int:item_id>', methods=['GET'])
def similar(item_id):
    engine = current_app.rec_engine
    content_type = request.args.get('type', 'movie')
    movies = engine.get_similar(item_id, content_type=content_type, n=12)
    return jsonify({'movies': movies})


@rec_bp.route('/personalized', methods=['GET'])
@jwt_required()
def personalized():
    engine  = current_app.rec_engine
    user_id = int(get_jwt_identity())

    m_ids, t_ids, m_w, t_w, excl_m, excl_t, ni_m, ni_t = _build_interest_profile(user_id)

    # Not-interested items are always excluded
    excl_m = list(set(excl_m) | set(ni_m))
    excl_t = list(set(excl_t) | set(ni_t))

    if not m_ids and not t_ids:
        return jsonify({'movies': engine.top_rated(n=24)})

    movies = engine.get_personalized(
        m_ids, t_ids, n=24,
        exclude_movie_ids=excl_m,
        exclude_tv_ids=excl_t,
        movie_weights=m_w,
        tv_weights=t_w,
    )
    return jsonify({'movies': movies})


@rec_bp.route('/surprise', methods=['GET'])
@jwt_required()
def surprise():
    engine  = current_app.rec_engine
    user_id = int(get_jwt_identity())

    m_ids, t_ids, m_w, t_w, excl_m, excl_t, ni_m, ni_t = _build_interest_profile(user_id)

    excl_m = list(set(excl_m) | set(ni_m))
    excl_t = list(set(excl_t) | set(ni_t))

    item = engine.get_surprise(
        m_ids or [], t_ids or [],
        exclude_movie_ids=excl_m,
        exclude_tv_ids=excl_t,
        movie_weights=m_w if m_ids else None,
        tv_weights=t_w if t_ids else None,
    )
    if not item:
        return jsonify({'movie': None}), 200
    return jsonify({'movie': item})


@rec_bp.route('/interact', methods=['POST'])
@jwt_required()
def log_interaction():
    """Log an implicit user interaction (detail view, search click, etc.)."""
    user_id = int(get_jwt_identity())
    data    = request.get_json()
    movie_id     = data.get('movie_id')
    content_type = data.get('content_type', 'movie')
    action       = data.get('action', 'view')

    if not movie_id:
        return jsonify({'error': 'movie_id required'}), 400

    ix = UserInteraction(
        user_id=user_id,
        movie_id=int(movie_id),
        content_type=content_type,
        action=action,
    )
    db.session.add(ix)
    db.session.commit()
    return jsonify({'ok': True}), 200


@rec_bp.route('/genre-stats', methods=['GET'])
@jwt_required()
def genre_stats():
    engine  = current_app.rec_engine
    user_id = int(get_jwt_identity())
    user_movies = UserMovie.query.filter_by(user_id=user_id).all()

    movie_ids = [um.movie_id for um in user_movies
                 if um.content_type == 'movie' and um.list_type in ('liked', 'watched', 'favorite')]
    tv_ids    = [um.movie_id for um in user_movies
                 if um.content_type == 'tv'    and um.list_type in ('liked', 'watched', 'favorite')]

    stats = engine.get_genre_stats(movie_ids, tv_ids)
    return jsonify({'genres': stats})


@rec_bp.route('/top', methods=['GET'])
def top_rated():
    engine = current_app.rec_engine
    content_type = request.args.get('type') or 'movie'
    movies = engine.top_rated(content_type=content_type, n=24)
    return jsonify({'movies': movies})
