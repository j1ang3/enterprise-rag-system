from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    """Declarative base shared by metadata models and Alembic."""
