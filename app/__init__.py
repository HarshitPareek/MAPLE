from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from app.config import Config

db = SQLAlchemy()
jwt = JWTManager()

def create_app():
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    app.config.from_object(Config)

    db.init_app(app)
    jwt.init_app(app)
    CORS(app)

    # Load ML engine
    from app.ml.engine import RecommendationEngine
    app.rec_engine = RecommendationEngine(
        app.config['PROCESSED_MOVIES_PATH'],
        app.config['SIMILARITY_MATRIX_PATH'],
        app.config['PROCESSED_TV_PATH'],
        app.config['SIMILARITY_MATRIX_TV_PATH'],
        movies_tfidf_path=app.config.get('TFIDF_MATRIX_MOVIES_PATH'),
        tv_tfidf_path=app.config.get('TFIDF_MATRIX_TV_PATH'),
        movies_collab_sim_path=app.config.get('COLLAB_SIM_MOVIES_PATH'),
    )

    # Register blueprints
    from app.routes.auth import auth_bp
    from app.routes.movies import movies_bp
    from app.routes.recommendations import rec_bp
    from app.routes.user_lists import lists_bp
    from app.routes.pages import pages_bp

    app.register_blueprint(pages_bp)
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(movies_bp, url_prefix='/api/movies')
    app.register_blueprint(rec_bp, url_prefix='/api/recommend')
    app.register_blueprint(lists_bp, url_prefix='/api/user')

    with app.app_context():
        db.create_all()

    return app
