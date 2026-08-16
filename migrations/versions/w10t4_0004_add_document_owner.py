"""Add nullable document ownership for explicit historical bootstrap.

Revision ID: w10t4_0004
Revises: w10t3_0003
Create Date: 2026-08-09
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "w10t4_0004"
down_revision: str | Sequence[str] | None = "w10t3_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("owner_id", sa.Uuid(), nullable=True),
    )
    op.create_foreign_key(
        "fk_documents_owner_id_users",
        "documents",
        "users",
        ["owner_id"],
        ["user_id"],
        ondelete="RESTRICT",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_documents_owner_id_users",
        "documents",
        type_="foreignkey",
    )
    op.drop_column("documents", "owner_id")
