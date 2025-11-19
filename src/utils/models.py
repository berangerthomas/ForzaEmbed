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
    pass

class Model(Base):
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
    __tablename__ = "generated_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(ForeignKey("models.name"), nullable=False)
    file_type: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    model: Mapped["Model"] = relationship("Model", back_populates="generated_files")

class GlobalChart(Base):
    __tablename__ = "global_charts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chart_type: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class ProcessingResult(Base):
    __tablename__ = "processing_results"
    __table_args__ = (UniqueConstraint("model_name", "file_id", name="uq_model_file"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String, nullable=False)
    file_id: Mapped[str] = mapped_column(String, nullable=False)
    results_blob: Mapped[bytes] = mapped_column(BLOB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class EmbeddingCache(Base):
    __tablename__ = "embedding_cache"
    
    # Composite primary key as defined in original schema
    model_name: Mapped[str] = mapped_column(String, primary_key=True)
    text_hash: Mapped[str] = mapped_column(String, primary_key=True)
    vector: Mapped[bytes] = mapped_column(BLOB, nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class TSNECoordinate(Base):
    __tablename__ = "tsne_coordinates"
    __table_args__ = (UniqueConstraint("tsne_key", "file_id", name="uq_tsne_file"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tsne_key: Mapped[str] = mapped_column(String, nullable=False)
    file_id: Mapped[str] = mapped_column(String, nullable=False)
    coordinates: Mapped[bytes] = mapped_column(BLOB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
