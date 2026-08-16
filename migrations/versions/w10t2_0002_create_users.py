"""Create the persistent user identity registry.

Revision ID: w10t2_0002
Revises: w10t1_0001
Create Date: 2026-08-09
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "w10t2_0002"
down_revision: str | Sequence[str] | None = "w10t1_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("user_id"),
        sa.UniqueConstraint("username", name="uq_users_username"),
    )


def downgrade() -> None:
    op.drop_table("users")
