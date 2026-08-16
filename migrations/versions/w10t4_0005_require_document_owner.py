"""Require every application document to have a valid owner.

Revision ID: w10t4_0005
Revises: w10t4_0004
Create Date: 2026-08-09
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "w10t4_0005"
down_revision: str | Sequence[str] | None = "w10t4_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    connection = op.get_bind()
    ownerless_count = connection.scalar(
        sa.text("SELECT count(*) FROM documents WHERE owner_id IS NULL")
    )
    if ownerless_count:
        raise RuntimeError(
            "Cannot require document ownership while ownerless documents remain; "
            "run the explicit ownership bootstrap first."
        )
    op.alter_column(
        "documents",
        "owner_id",
        existing_type=sa.Uuid(),
        nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "documents",
        "owner_id",
        existing_type=sa.Uuid(),
        nullable=True,
    )
