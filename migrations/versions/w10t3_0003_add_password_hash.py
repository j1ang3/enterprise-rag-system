"""Add one-way password credentials to users.

Revision ID: w10t3_0003
Revises: w10t2_0002
Create Date: 2026-08-09
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "w10t3_0003"
down_revision: str | Sequence[str] | None = "w10t2_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    connection = op.get_bind()
    existing_user_count = connection.scalar(sa.text("SELECT count(*) FROM users"))
    if existing_user_count:
        raise RuntimeError(
            "Cannot add a required password_hash while users already exist; "
            "credential transition must be reassessed without fabricating passwords."
        )
    op.add_column(
        "users",
        sa.Column("password_hash", sa.Text(), nullable=False),
    )


def downgrade() -> None:
    op.drop_column("users", "password_hash")
