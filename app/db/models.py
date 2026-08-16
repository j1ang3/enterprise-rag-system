from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class DocumentRecord(Base):
    """Persistent application metadata for one indexed document."""

    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(
            "file_size_bytes IS NULL OR file_size_bytes >= 0",
            name="ck_documents_file_size_nonnegative",
        ),
        CheckConstraint(
            "chunk_count >= 0",
            name="ck_documents_chunk_count_nonnegative",
        ),
    )

    # Historical evaluation documents use human-readable IDs, so this must not
    # be a native PostgreSQL UUID even though new uploads currently use UUIDs.
    document_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_extension: Mapped[str] = mapped_column(String(32), nullable=False)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    upload_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    text_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    owner_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey(
            "users.user_id",
            name="fk_documents_owner_id_users",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )


class UserRecord(Base):
    """Persistent user identity and one-way password credential."""

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("username", name="uq_users_username"),
    )

    user_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    username: Mapped[str] = mapped_column(String(64), nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )


class DocumentACLRecord(Base):
    """One explicit non-owner read grant for an application document."""

    __tablename__ = "document_acl"
    __table_args__ = (
        Index("ix_document_acl_user_id", "user_id"),
    )

    document_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey(
            "documents.document_id",
            name="fk_document_acl_document_id_documents",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey(
            "users.user_id",
            name="fk_document_acl_user_id_users",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
