from flask import Blueprint, render_template, request

pages_bp = Blueprint('pages', __name__)

@pages_bp.route('/')
def index():
    return render_template('index.html')

@pages_bp.route('/auth')
def auth():
    return render_template('auth.html')

@pages_bp.route('/movie/<int:movie_id>')
def movie_detail(movie_id):
    content_type = request.args.get('type', 'movie')
    return render_template('movie_detail.html', movie_id=movie_id, content_type=content_type)

@pages_bp.route('/profile')
def profile():
    return render_template('profile.html')

@pages_bp.route('/watchlist')
def watchlist():
    return render_template('watchlist.html')
