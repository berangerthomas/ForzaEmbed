"""SQLAlchemy ORM models for the ForzaEmbed database.

This module defines all database models used for storing embedding results,
metrics, and metadata. Uses SQLAlchemy 2.0 declarative mapping with type
annotations.

Models:
    Model: Stores model run configurations.
    EvaluationMetric: Stores evaluation metrics for each model run.
    GeneratedFile: Tracks generated output files.
    GlobalChart: Stores paths to global chart images.
    ProcessingResult: Stores detailed processing results per file.
    EmbeddingCache: Caches computed embeddings for reuse.
    TSNECoordinate: Caches t-SNE coordinate calculations.
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
        metrics: Related evaluation metrics.
        generated_files: Related generated files.
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
    metrics: Mapped["EvaluationMetric"] = relationship(
        "EvaluationMetric", back_populates="model", cascade="all, delete-orphan"
    )
    generated_files: Mapped[list["GeneratedFile"]] = relationship(
        "GeneratedFile", back_populates="model", cascade="all, delete-orphan"
    )


class EvaluationMetric(Base):
    """Stores evaluation metrics for a model run.

    Attributes:
        id: Primary key.
        model_name: Foreign key to the model.
        silhouette_score: Silhouette clustering score.
        intra_cluster_distance_normalized: Normalized intra-cluster distance.
        inter_cluster_distance_normalized: Normalized inter-cluster distance.
        embedding_computation_time: Time taken to compute embeddings.
        created_at: Timestamp of creation.
        model: Related model instance.
    """

    __tablename__ = "evaluation_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(ForeignKey("models.name"), nullable=False)
    silhouette_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    intra_cluster_distance_normalized: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    inter_cluster_distance_normalized: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    embedding_computation_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    model: Mapped["Model"] = relationship("Model", back_populates="metrics")


class GeneratedFile(Base):
    """Tracks generated output files for a model run.

    Attributes:
        id: Primary key.
        model_name: Foreign key to the model.
        file_type: Type of the generated file.
        file_path: Path to the generated file.
        created_at: Timestamp of creation.
        model: Related model instance.
    """

    __tablename__ = "generated_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(ForeignKey("models.name"), nullable=False)
    file_type: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    model: Mapped["Model"] = relationship("Model", back_populates="generated_files")


class GlobalChart(Base):
    """Stores paths to global chart images.

    Attributes:
        id: Primary key.
        chart_type: Type identifier for the chart.
        file_path: Path to the chart image file.
        created_at: Timestamp of creation.
    """

    __tablename__ = "global_charts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chart_type: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


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


class TSNECoordinate(Base):
    """Caches t-SNE coordinate calculations.

    Attributes:
        id: Primary key.
        tsne_key: Unique key for the t-SNE configuration.
        file_id: Identifier for the file.
        coordinates: Serialized coordinate data.
        created_at: Timestamp of creation.
    """

    __tablename__ = "tsne_coordinates"
    __table_args__ = (UniqueConstraint("tsne_key", "file_id", name="uq_tsne_file"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tsne_key: Mapped[str] = mapped_column(String, nullable=False)
    file_id: Mapped[str] = mapped_column(String, nullable=False)
    coordinates: Mapped[bytes] = mapped_column(BLOB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
