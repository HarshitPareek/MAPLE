"""Model package.

These imports look unused but are required: importing them registers each
model with SQLAlchemy's metadata so ``db.create_all()`` creates the tables.
"""

from .user import User
from .user_movie import UserMovie
from .user_interaction import UserInteraction

__all__ = ['User', 'UserMovie', 'UserInteraction']
