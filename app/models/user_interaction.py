from app import db
from app.utils import utcnow

class UserInteraction(db.Model):
    __tablename__ = 'user_interactions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    movie_id = db.Column(db.Integer, nullable=False)
    content_type = db.Column(db.String(10), default='movie')
    action = db.Column(db.String(20), nullable=False)  # 'view', 'detail_open', 'search_click'
    timestamp = db.Column(db.DateTime, default=utcnow)
