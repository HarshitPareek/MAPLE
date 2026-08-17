from flask import Blueprint, request, jsonify, current_app
from app.utils import parse_int

movies_bp = Blueprint('movies', __name__)

@movies_bp.route('/', methods=['GET'])
def get_movies():
    engine = current_app.rec_engine
    # A non-numeric or out-of-range page falls back to page 1 rather than
    # raising; page < 1 would otherwise slice the catalog from the tail.
    page = parse_int(request.args.get('page'), default=1, minimum=1)
    genre = request.args.get('genre') or None
    content_type = request.args.get('type') or None  # 'movie', 'tv', or None (all)
    return jsonify(engine.get_all(page=page, per_page=24, genre=genre, content_type=content_type))

@movies_bp.route('/search', methods=['GET'])
def search():
    engine = current_app.rec_engine
    query = request.args.get('q', '').strip()
    content_type = request.args.get('type') or None
    if not query:
        return jsonify({'movies': []})
    results = engine.search(query, content_type=content_type, limit=24)
    return jsonify({'movies': results})

@movies_bp.route('/autocomplete', methods=['GET'])
def autocomplete():
    engine = current_app.rec_engine
    query = request.args.get('q', '').strip()
    content_type = request.args.get('type') or None
    if not query or len(query) < 2:
        return jsonify({'suggestions': []})
    suggestions = engine.autocomplete(query, content_type=content_type, limit=8)
    return jsonify({'suggestions': suggestions})

@movies_bp.route('/<int:movie_id>', methods=['GET'])
def get_movie(movie_id):
    engine = current_app.rec_engine
    content_type = request.args.get('type', 'movie')
    item = engine.get_item(movie_id, content_type)
    if not item:
        return jsonify({'error': 'Not found'}), 404
    return jsonify({'movie': item})

@movies_bp.route('/genres', methods=['GET'])
def get_genres():
    engine = current_app.rec_engine
    content_type = request.args.get('type') or None
    return jsonify({'genres': engine.get_genres(content_type)})


