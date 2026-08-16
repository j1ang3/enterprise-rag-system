"""Create explicit document read grants.

Revision ID: w10t5_0006
Revises: w10t4_0005
Create Date: 2026-08-09
"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "w10t5_0006"
down_revision: str | Sequence[str] | None = "w10t4_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "document_acl",
        sa.Column("document_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.document_id"],
            name="fk_document_acl_document_id_documents",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.user_id"],
            name="fk_document_acl_user_id_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "document_id",
            "user_id",
            name="pk_document_acl",
        ),
    )
    op.create_index(
        "ix_document_acl_user_id",
        "document_acl",
        ["user_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_document_acl_user_id", table_name="document_acl")
    op.drop_table("document_acl")
