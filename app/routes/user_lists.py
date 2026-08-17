from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import db
from app.models.user_movie import UserMovie
from app.utils import parse_item_id
from sqlalchemy.exc import IntegrityError

VALID_CONTENT_TYPES = ('movie', 'tv')

lists_bp = Blueprint('lists', __name__)

def _get_list(list_type):
    user_id = int(get_jwt_identity())
    items = UserMovie.query.filter_by(user_id=user_id, list_type=list_type).all()
    return jsonify({
        'movie_ids': [i.movie_id for i in items if i.content_type == 'movie'],
        'tv_ids':    [i.movie_id for i in items if i.content_type == 'tv'],
    })

def _add(list_type):
    user_id = int(get_jwt_identity())
    data = request.get_json(silent=True) or {}
    movie_id = parse_item_id(data.get('movie_id'))
    content_type = data.get('content_type', 'movie')
    if movie_id is None:
        return jsonify({'error': 'a valid movie_id is required'}), 400
    if content_type not in VALID_CONTENT_TYPES:
        return jsonify({'error': "content_type must be 'movie' or 'tv'"}), 400
    entry = UserMovie(user_id=user_id, movie_id=movie_id, list_type=list_type, content_type=content_type)
    db.session.add(entry)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
    return jsonify({'ok': True}), 200

def _remove(list_type, movie_id, content_type):
    user_id = int(get_jwt_identity())
    UserMovie.query.filter_by(
        user_id=user_id, movie_id=int(movie_id),
        list_type=list_type, content_type=content_type
    ).delete()
    db.session.commit()
    return jsonify({'ok': True}), 200

@lists_bp.route('/favorites', methods=['GET'])
@jwt_required()
def get_favorites():
    return _get_list('favorite')

@lists_bp.route('/favorites', methods=['POST'])
@jwt_required()
def add_favorite():
    return _add('favorite')

@lists_bp.route('/favorites/<int:movie_id>', methods=['DELETE'])
@jwt_required()
def remove_favorite(movie_id):
    content_type = request.args.get('type', 'movie')
    return _remove('favorite', movie_id, content_type)

@lists_bp.route('/watched', methods=['GET'])
@jwt_required()
def get_watched():
    return _get_list('watched')

@lists_bp.route('/watched', methods=['POST'])
@jwt_required()
def add_watched():
    return _add('watched')

@lists_bp.route('/watched/<int:movie_id>', methods=['DELETE'])
@jwt_required()
def remove_watched(movie_id):
    content_type = request.args.get('type', 'movie')
    return _remove('watched', movie_id, content_type)

@lists_bp.route('/liked', methods=['GET'])
@jwt_required()
def get_liked():
    return _get_list('liked')

@lists_bp.route('/liked', methods=['POST'])
@jwt_required()
def add_liked():
    return _add('liked')

@lists_bp.route('/liked/<int:movie_id>', methods=['DELETE'])
@jwt_required()
def remove_liked(movie_id):
    content_type = request.args.get('type', 'movie')
    return _remove('liked', movie_id, content_type)

@lists_bp.route('/watchlist', methods=['GET'])
@jwt_required()
def get_watchlist():
    return _get_list('watchlist')

@lists_bp.route('/watchlist', methods=['POST'])
@jwt_required()
def add_watchlist():
    return _add('watchlist')

@lists_bp.route('/watchlist/<int:movie_id>', methods=['DELETE'])
@jwt_required()
def remove_watchlist(movie_id):
    content_type = request.args.get('type', 'movie')
    return _remove('watchlist', movie_id, content_type)

@lists_bp.route('/not-interested', methods=['GET'])
@jwt_required()
def get_not_interested():
    return _get_list('not_interested')

@lists_bp.route('/not-interested', methods=['POST'])
@jwt_required()
def add_not_interested():
    return _add('not_interested')

@lists_bp.route('/not-interested/<int:movie_id>', methods=['DELETE'])
@jwt_required()
def remove_not_interested(movie_id):
    content_type = request.args.get('type', 'movie')
    return _remove('not_interested', movie_id, content_type)
