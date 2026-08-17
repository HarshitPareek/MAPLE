from app import db
from app.utils import utcnow

class UserMovie(db.Model):
    __tablename__ = 'user_movies'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    movie_id = db.Column(db.Integer, nullable=False)
    list_type = db.Column(db.String(20), nullable=False)      # 'favorite' or 'watched'
    content_type = db.Column(db.String(10), default='movie')  # 'movie' or 'tv'
    added_at = db.Column(db.DateTime, default=utcnow)
    __table_args__ = (
        db.UniqueConstraint('user_id', 'movie_id', 'list_type', 'content_type', name='unique_user_item_list'),
    )

    def to_dict(self):
        return {'id': self.id, 'movie_id': self.movie_id, 'list_type': self.list_type, 'content_type': self.content_type}
