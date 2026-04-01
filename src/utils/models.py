"""SQLAlchemy ORM models for the ForzaEmbed database.

This module defines all database models used for storing embedding results,
metrics, and metadata. Uses SQLAlchemy 2.0 declarative mapping with type
annotations.

Models:
    Model: Stores model run configurations.
    ProcessingResult: Stores detailed processing results per file.
    EmbeddingCache: Caches computed embeddings for reuse.
    ProjectionCoordinate: Caches dimensional reduction coordinates.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BLOB,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""

    pass


class Model(Base):
    """Stores model run configuration and metadata.

    Attributes:
        id: Primary key.
        name: Unique run name identifier.
        base_model_name: The underlying model name.
        type: Model type (api, fastembed, sentence_transformers, etc.).
        chunk_size: Chunk size used in this run.
        chunk_overlap: Chunk overlap used in this run.
        theme_name: Theme set name used.
        chunking_strategy: Chunking strategy used.
        similarity_metric: Similarity metric used.
        created_at: Timestamp of creation.
                    """

    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    base_model_name: Mapped[str] = mapped_column(String, nullable=False)
    type: Mapped[str] = mapped_column(String, nullable=False)
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_overlap: Mapped[int] = mapped_column(Integer, nullable=False)
    theme_name: Mapped[str] = mapped_column(String, nullable=False)
    chunking_strategy: Mapped[str] = mapped_column(String, nullable=False)
    similarity_metric: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships


class ProcessingResult(Base):
    """Stores detailed processing results for each file.

    Attributes:
        id: Primary key.
        model_name: The model run name.
        file_id: Identifier for the processed file.
        results_blob: Serialized results data.
        created_at: Timestamp of creation.
    """

    __tablename__ = "processing_results"
    __table_args__ = (UniqueConstraint("model_name", "file_id", name="uq_model_file"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    file_id: Mapped[str] = mapped_column(String, nullable=False)
    results_blob: Mapped[bytes] = mapped_column(BLOB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class EmbeddingCache(Base):
    """Caches computed embeddings for reuse.

    Attributes:
        model_name: The model name (part of composite primary key).
        text_hash: Hash of the embedded text (part of composite primary key).
        vector: Serialized embedding vector.
        dimension: Dimension of the embedding vector.
        created_at: Timestamp of creation.
    """

    __tablename__ = "embedding_cache"

    # Composite primary key as defined in original schema
    model_name: Mapped[str] = mapped_column(String, primary_key=True)
    text_hash: Mapped[str] = mapped_column(String, primary_key=True)
    vector: Mapped[bytes] = mapped_column(BLOB, nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ProjectionCoordinate(Base):
    """Caches dimensional reduction coordinate calculations (t-SNE, UMAP, PCA).

    Attributes:
        id: Primary key.
        projection_key: Unique key for the projection configuration.
        file_id: Identifier for the file.
        coordinates: Serialized coordinate data.
        created_at: Timestamp of creation.
    """

    __tablename__ = "projection_coordinates"
    __table_args__ = (UniqueConstraint("projection_key", "file_id", name="uq_projection_file"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    projection_key: Mapped[str] = mapped_column(String, nullable=False)
    file_id: Mapped[str] = mapped_column(String, nullable=False)
    coordinates: Mapped[bytes] = mapped_column(BLOB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
