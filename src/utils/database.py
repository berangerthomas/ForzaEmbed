"""Database management module for ForzaEmbed.

This module provides the EmbeddingDatabase class for managing all database
operations including storing embeddings, results, and metadata. It implements
intelligent quantization for efficient storage and caching mechanisms for
improved performance.

Example:
    Basic database usage::

        from src.utils.database import EmbeddingDatabase

        db = EmbeddingDatabase("results.db", config)
        db.save_embeddings_batch("model_name", embeddings_dict)
        cached = db.get_embeddings_by_hashes("model_name", ["hash1", "hash2"])
"""

import logging
import os
import zlib
from typing import Any, Dict, List, Optional, Tuple, Union

import msgpack
import numpy as np
from sqlalchemy import create_engine, delete, select, text, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import sessionmaker

from ..core.config import AppConfig
from .models import (
    Base,
    EmbeddingCache,
    EvaluationMetric,
    GeneratedFile,
    GlobalChart,
    Model,
    ProcessingResult,
    TSNECoordinate,
)


class EmbeddingDatabase:
    """Manage SQLite database for embeddings, results, and metadata.

    This class handles all database operations for ForzaEmbed, including
    storing and retrieving embeddings, processing results, and various
    metadata. Implements intelligent quantization to reduce storage size.

    Attributes:
        db_path: Path to the SQLite database file.
        config: Application configuration (dict or AppConfig).
        quantization_enabled: Whether intelligent quantization is enabled.
        engine: SQLAlchemy database engine.
        Session: SQLAlchemy session factory.
    """

    def __init__(
        self, db_path: str, config: Union[AppConfig, Dict[str, Any]]
    ) -> None:
        """Initialize the EmbeddingDatabase.

        Args:
            db_path: Path to the SQLite database file.
            config: Application configuration, either as AppConfig or dict.
        """
        self.db_path = db_path
        self.config = config
        if isinstance(config, dict):
            self.quantization_enabled: bool = config.get("database", {}).get(
                "intelligent_quantization", True
            )
            self.quantize_metrics: bool = config.get("database", {}).get(
                "quantize_metrics", True
            )
        else:
            self.quantization_enabled = config.database.intelligent_quantization
            self.quantize_metrics = config.database.quantize_metrics

        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        # Initialize SQLAlchemy engine and session
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.Session = sessionmaker(bind=self.engine)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the database tables."""
        Base.metadata.create_all(self.engine)

    def add_model(
        self,
        name: str,
        base_model_name: str,
        model_type: str,
        chunk_size: int,
        chunk_overlap: int,
        theme_name: str,
        chunking_strategy: str,
        similarity_metric: str,
    ) -> None:
        """Add a model run to the database.

        Args:
            name: Unique run name identifier.
            base_model_name: The underlying model name.
            model_type: Type of model (api, fastembed, etc.).
            chunk_size: Chunk size used.
            chunk_overlap: Chunk overlap used.
            theme_name: Theme set name.
            chunking_strategy: Chunking strategy used.
            similarity_metric: Similarity metric used.
        """
        with self.Session() as session:
            stmt = sqlite_insert(Model).values(
                name=name,
                base_model_name=base_model_name,
                type=model_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                theme_name=theme_name,
                chunking_strategy=chunking_strategy,
                similarity_metric=similarity_metric,
            ).on_conflict_do_nothing()
            session.execute(stmt)
            session.commit()

    def add_evaluation_metrics(
        self, model_name: str, metrics: Dict[str, float]
    ) -> None:
        """Add or update evaluation metrics for a model.

        Args:
            model_name: The model run name.
            metrics: Dictionary of metric names to values.
        """
        with self.Session() as session:
            # Delete existing metrics for this model
            session.execute(
                delete(EvaluationMetric).where(EvaluationMetric.model_name == model_name)
            )

            # Insert new metrics
            metric = EvaluationMetric(
                model_name=model_name,
                silhouette_score=metrics.get("silhouette_score"),
                intra_cluster_distance_normalized=metrics.get("intra_cluster_distance_normalized"),
                inter_cluster_distance_normalized=metrics.get("inter_cluster_distance_normalized"),
                embedding_computation_time=metrics.get("embedding_computation_time"),
            )
            session.add(metric)
            session.commit()

    def add_generated_file(
        self, model_name: str, file_type: str, file_path: str
    ) -> None:
        """Add a generated file record to the database.

        Args:
            model_name: The model run name.
            file_type: Type of the generated file.
            file_path: Path to the generated file.
        """
        with self.Session() as session:
            file_entry = GeneratedFile(
                model_name=model_name,
                file_type=file_type,
                file_path=file_path
            )
            session.add(file_entry)
            session.commit()

    def add_global_chart(self, chart_type: str, file_path: str) -> None:
        """Add or update a global chart record.

        Args:
            chart_type: Type identifier for the chart.
            file_path: Path to the chart image file.
        """
        with self.Session() as session:
            # Delete existing chart of this type
            session.execute(
                delete(GlobalChart).where(GlobalChart.chart_type == chart_type)
            )
            # Insert new chart
            chart = GlobalChart(chart_type=chart_type, file_path=file_path)
            session.add(chart)
            session.commit()

    def model_exists(self, name: str) -> bool:
        """Check if a model with the specified run name exists.

        Args:
            name: The run name to check.

        Returns:
            True if the model exists, False otherwise.
        """
        with self.Session() as session:
            result = session.execute(
                select(Model).where(Model.name == name)
            ).first()
            return result is not None

    def save_processing_result(
        self, model_name: str, file_id: str, results: Dict[str, Any]
    ) -> None:
        """Save detailed processing result for a file and model.

        Args:
            model_name: The model run name.
            file_id: Identifier for the processed file.
            results: Dictionary of processing results.
        """
        # Apply intelligent quantization before serialization
        quantized_results = self._apply_intelligent_quantization(results)
        results_blob = msgpack.packb(
            quantized_results, default=self._numpy_default, use_bin_type=True
        )

        with self.Session() as session:
            stmt = sqlite_insert(ProcessingResult).values(
                model_name=model_name,
                file_id=file_id,
                results_blob=results_blob
            )
            # Upsert: replace if exists
            stmt = stmt.on_conflict_do_update(
                index_elements=['model_name', 'file_id'],
                set_=dict(results_blob=results_blob)
            )
            session.execute(stmt)
            session.commit()

    def save_processing_results_batch(
        self, results_batch: List[Tuple[str, str, Dict[str, Any]]]
    ) -> None:
        """Save a batch of processing results in a single transaction.

        Args:
            results_batch: List of (model_name, file_id, results) tuples.
        """
        items_to_insert = []
        for model_name, file_id, results in results_batch:
            # Apply intelligent quantization before serialization
            quantized_results = self._apply_intelligent_quantization(results)
            results_blob = msgpack.packb(
                quantized_results, default=self._numpy_default, use_bin_type=True
            )
            items_to_insert.append({
                "model_name": model_name,
                "file_id": file_id,
                "results_blob": results_blob
            })

        if not items_to_insert:
            return

        with self.Session() as session:
            stmt = sqlite_insert(ProcessingResult).values(items_to_insert)
            stmt = stmt.on_conflict_do_update(
                index_elements=['model_name', 'file_id'],
                set_=dict(results_blob=stmt.excluded.results_blob)
            )
            session.execute(stmt)
            session.commit()

    def get_processed_files(self, model_name: str) -> List[str]:
        """Retrieve file IDs that have been processed for a model.

        Args:
            model_name: The model run name.

        Returns:
            List of file IDs that have been processed.
        """
        with self.Session() as session:
            result = session.execute(
                select(ProcessingResult.file_id).where(ProcessingResult.model_name == model_name)
            )
            return [row[0] for row in result.fetchall()]

    def get_model_info(self, run_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve information about a model by its run name.

        Args:
            run_name: The unique run name identifier.

        Returns:
            Dictionary with model information, or None if not found.
        """
        with self.Session() as session:
            model = session.execute(
                select(Model).where(Model.name == run_name)
            ).scalar_one_or_none()
            
            if model:
                return {
                    "model_name": model.name,
                    "type": model.type,
                    "chunk_size": model.chunk_size,
                    "chunk_overlap": model.chunk_overlap,
                    "theme_name": model.theme_name,
                    "chunking_strategy": model.chunking_strategy,
                }
            return None

    def get_all_processing_results(self) -> Dict[str, Any]:
        """Retrieve all processing results organized by model run name.

        Fetches raw file-level results without model-level aggregation.

        Returns:
            Dictionary mapping model names to their file results.
        """
        all_results: Dict[str, Dict[str, Any]] = {}
        with self.Session() as session:
            # Get all unique model names first
            model_names = session.execute(
                select(ProcessingResult.model_name).distinct()
            ).scalars().all()

            for model_name in model_names:
                # Get all file results for the current model
                file_results = self.get_all_processing_results_for_run(model_name)
                all_results[model_name] = {"files": file_results}
        return all_results

    def get_all_models(self) -> List[Dict[str, Any]]:
        """Retrieve all models with their metrics.

        Returns:
            List of dictionaries containing model information and metrics.
        """
        with self.Session() as session:
            # Left join Model and EvaluationMetric
            stmt = select(Model, EvaluationMetric).outerjoin(
                EvaluationMetric, Model.name == EvaluationMetric.model_name
            ).order_by(Model.name)
            
            results = session.execute(stmt).all()
            
            models = []
            for model, metric in results:
                models.append(
                    {
                        "name": model.name,
                        "base_model_name": model.base_model_name,
                        "type": model.type,
                        "chunk_size": model.chunk_size,
                        "chunk_overlap": model.chunk_overlap,
                        "theme_name": model.theme_name,
                        "chunking_strategy": model.chunking_strategy,
                        "silhouette_score": metric.silhouette_score if metric else None,
                        "intra_cluster_distance_normalized": metric.intra_cluster_distance_normalized if metric else None,
                        "inter_cluster_distance_normalized": metric.inter_cluster_distance_normalized if metric else None,
                        "embedding_computation_time": metric.embedding_computation_time if metric else None,
                    }
                )

            return models

    def get_model_files(self, model_name: str) -> List[Dict[str, str]]:
        """Retrieve all generated files for a model.

        Args:
            model_name: The model run name.

        Returns:
            List of dictionaries with file type and path.
        """
        with self.Session() as session:
            files = session.execute(
                select(GeneratedFile).where(GeneratedFile.model_name == model_name).order_by(GeneratedFile.file_type)
            ).scalars().all()

            return [{"type": f.file_type, "path": f.file_path} for f in files]

    def get_global_charts(self) -> List[Dict[str, str]]:
        """Retrieve all global charts.

        Returns:
            List of dictionaries with chart type and path.
        """
        with self.Session() as session:
            charts = session.execute(
                select(GlobalChart).order_by(GlobalChart.chart_type)
            ).scalars().all()

            return [{"type": c.chart_type, "path": c.file_path} for c in charts]

    def vacuum_database(self) -> None:
        """Vacuum the database to reclaim space."""
        with self.Session() as session:
            session.execute(text("VACUUM"))

    def get_all_run_names(self) -> list[str]:
        """Retrieve all existing run names.

        Returns:
            List of unique run name identifiers.
        """
        with self.Session() as session:
            return session.execute(
                select(Model.name).distinct()
            ).scalars().all()

    def get_processed_files_with_similarities(self, run_name: str) -> list[str]:
        """Retrieve files that have been processed with similarity scores.

        Args:
            run_name: The model run name.

        Returns:
            List of file IDs that have similarity data.
        """
        processed_files = []
        with self.Session() as session:
            results = session.execute(
                select(ProcessingResult.file_id, ProcessingResult.results_blob)
                .where(ProcessingResult.model_name == run_name)
            ).all()
            
            for file_id, results_blob in results:
                results_data = msgpack.unpackb(
                    results_blob, object_hook=self._decode_numpy, raw=False
                )
                if "similarities" in results_data and results_data["similarities"] is not None:
                    processed_files.append(file_id)
        return processed_files

    def get_embeddings_by_hashes(
        self, base_model_name: str, text_hashes: List[str]
    ) -> Dict[str, np.ndarray]:
        """Retrieve embeddings from cache by model and text hashes.

        Args:
            base_model_name: The base model name used for embeddings.
            text_hashes: List of text hash values to retrieve.

        Returns:
            Dictionary mapping text hashes to embedding arrays.
        """
        if not text_hashes:
            return {}

        with self.Session() as session:
            # SQLAlchemy IN clause
            results = session.execute(
                select(EmbeddingCache.text_hash, EmbeddingCache.vector)
                .where(EmbeddingCache.model_name == base_model_name)
                .where(EmbeddingCache.text_hash.in_(text_hashes))
            ).all()

            embeddings = {}
            for text_hash, vector_blob in results:
                embeddings[text_hash] = msgpack.unpackb(
                    vector_blob, object_hook=self._decode_numpy, raw=False
                )
            return embeddings

    def save_embeddings_batch(
        self, base_model_name: str, embeddings: Dict[str, np.ndarray]
    ) -> None:
        """Save a batch of embeddings to the cache.

        Applies intelligent quantization to reduce storage size when enabled.

        Args:
            base_model_name: The base model name for the embeddings.
            embeddings: Dictionary mapping text hashes to embedding arrays.
        """
        if not embeddings:
            return

        # Flag for maximum compression in cache
        self._cache_storage = True

        items_to_insert = []
        for text_hash, vector in embeddings.items():
            # Apply intelligent quantization to embeddings for cache storage
            if self.quantization_enabled and vector.dtype.kind == "f":
                # Most embeddings are normalized [-1,1] → float16 is sufficient
                if -1.1 <= vector.min() and vector.max() <= 1.1:
                    vector = vector.astype(np.float16)
                elif vector.dtype == np.float64:
                    # Downcast float64 → float32
                    vector = vector.astype(np.float32)

            vector_blob = msgpack.packb(
                vector, default=self._numpy_default, use_bin_type=True
            )
            dimension = len(vector) if len(vector.shape) == 1 else vector.shape[1]
            items_to_insert.append({
                "model_name": base_model_name,
                "text_hash": text_hash,
                "vector": vector_blob,
                "dimension": dimension
            })

        if hasattr(self, "_cache_storage"):
            delattr(self, "_cache_storage")

        with self.Session() as session:
            # Pass items as the second argument to trigger SQLAlchemy's executemany
            # mode, which avoids the SQLite 999-variable limit that occurs when
            # using .values(large_list) which inlines all parameters in one statement.
            stmt = sqlite_insert(EmbeddingCache).on_conflict_do_nothing()
            session.execute(stmt, items_to_insert)
            session.commit()

    def save_tsne_coordinates(
        self, tsne_key: str, file_id: str, coordinates: Dict[str, List[float]]
    ) -> None:
        """Save t-SNE coordinates for a given configuration.

        Args:
            tsne_key: Unique key for the t-SNE configuration.
            file_id: Identifier for the file.
            coordinates: Dictionary with 'x' and 'y' coordinate lists.
        """
        coordinates_blob = msgpack.packb(coordinates, use_bin_type=True)

        with self.Session() as session:
            stmt = sqlite_insert(TSNECoordinate).values(
                tsne_key=tsne_key,
                file_id=file_id,
                coordinates=coordinates_blob
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=['tsne_key', 'file_id'],
                set_=dict(coordinates=coordinates_blob)
            )
            session.execute(stmt)
            session.commit()

    def get_tsne_coordinates(
        self, tsne_key: str, file_id: str
    ) -> Optional[Dict[str, List[float]]]:
        """Retrieve t-SNE coordinates for a given configuration.

        Args:
            tsne_key: Unique key for the t-SNE configuration.
            file_id: Identifier for the file.

        Returns:
            Dictionary with 'x' and 'y' coordinate lists, or None if not found.
        """
        with self.Session() as session:
            result = session.execute(
                select(TSNECoordinate.coordinates)
                .where(TSNECoordinate.tsne_key == tsne_key)
                .where(TSNECoordinate.file_id == file_id)
            ).scalar_one_or_none()
            
            if result:
                return msgpack.unpackb(result, raw=False)
            return None

    def clear_tsne_cache(self) -> None:
        """Clear all cached t-SNE coordinates."""
        with self.Session() as session:
            session.execute(delete(TSNECoordinate))
            session.commit()

    def get_run_details(self, run_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve detailed information for a specific run.

        Args:
            run_name: The unique run name identifier.

        Returns:
            Dictionary with full run details, or None if not found.
        """
        with self.Session() as session:
            model = session.execute(
                select(Model).where(Model.name == run_name)
            ).scalar_one_or_none()
            
            if model:
                return {
                    "id": model.id,
                    "name": model.name,
                    "base_model_name": model.base_model_name,
                    "type": model.type,
                    "chunk_size": model.chunk_size,
                    "chunk_overlap": model.chunk_overlap,
                    "theme_name": model.theme_name,
                    "chunking_strategy": model.chunking_strategy,
                    "similarity_metric": model.similarity_metric,
                    "created_at": model.created_at
                }
            return None

    def get_all_processing_results_for_run(
        self, model_name: str
    ) -> Dict[str, Dict[str, Any]]:
        """Get all processing results for a specific run with dequantization.

        Args:
            model_name: The model run name.

        Returns:
            Dictionary mapping file IDs to their processing results.
        """
        results = {}
        with self.Session() as session:
            query_results = session.execute(
                select(ProcessingResult.file_id, ProcessingResult.results_blob)
                .where(ProcessingResult.model_name == model_name)
            ).all()
            
            for file_id, results_blob in query_results:
                data = msgpack.unpackb(
                    results_blob, object_hook=self._decode_numpy, raw=False
                )
                # Restore quantized data
                restored_data = self._restore_quantized_data(data)
                # Convert all numpy types to native Python types for JSON serialization
                results[file_id] = self._to_native_python_types(restored_data)
        return results

    def _to_native_python_types(self, obj: Any) -> Any:
        """Recursively convert NumPy types to native Python types.

        Args:
            obj: Object potentially containing NumPy types.

        Returns:
            Object with all NumPy types converted to Python native types.
        """
        if isinstance(obj, dict):
            return {k: self._to_native_python_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_native_python_types(i) for i in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Use abstract base classes for broad compatibility (including NumPy 2.0)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def _dequantize_similarities(
        self, similarities: np.ndarray | list[float]
    ) -> list[float]:
        """Dequantize similarities to [0,1] range.

        Args:
            similarities: Quantized or raw similarity values.

        Returns:
            List of dequantized similarity values in [0,1] range.
        """
        if isinstance(similarities, list):
            similarities = np.array(similarities)

        # If values are uint16 (quantized), convert back to float [0,1]
        if similarities.dtype == np.uint16:
            similarities = similarities.astype(np.float32) / 65535.0
        elif similarities.dtype in [np.int32, np.int64]:
            # Handle case where uint16 was read as int
            similarities = np.clip(similarities, 0, 65535).astype(np.float32) / 65535.0

        # Ensure values are in [0,1] range
        similarities = np.clip(similarities, 0.0, 1.0)
        return similarities.tolist()

    def update_metrics_for_file(
        self, model_name: str, file_id: str, metrics: Dict[str, Any]
    ) -> None:
        """Update metrics for a specific file in a model run.

        Args:
            model_name: The model run name.
            file_id: Identifier for the file.
            metrics: Dictionary of metric values to update.
        """
        with self.Session() as session:
            # First, retrieve the existing blob
            result = session.execute(
                select(ProcessingResult.results_blob)
                .where(ProcessingResult.model_name == model_name)
                .where(ProcessingResult.file_id == file_id)
            ).scalar_one_or_none()
            
            if not result:
                return  # Or handle error

            # Deserialize, update, and re-serialize
            results_data = msgpack.unpackb(
                result, object_hook=self._decode_numpy, raw=False
            )
            # Restore quantized data before update
            results_data = self._restore_quantized_data(results_data)
            results_data["metrics"].update(metrics)

            # Apply quantization and serialize
            quantized_results = self._apply_intelligent_quantization(results_data)
            updated_blob = msgpack.packb(
                quantized_results, default=self._numpy_default, use_bin_type=True
            )

            # Update the blob in the database
            stmt = update(ProcessingResult).where(
                ProcessingResult.model_name == model_name,
                ProcessingResult.file_id == file_id
            ).values(results_blob=updated_blob)
            
            session.execute(stmt)
            session.commit()

    def _quantize_metrics(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Quantize metrics based on their expected ranges.

        Applies intelligent quantization to reduce storage size while
        maintaining precision for the metric's expected range.

        Args:
            metrics: Dictionary of metric names to values.

        Returns:
            Dictionary with quantized metric values.
        """
        if not self.quantization_enabled:
            return metrics

        quantized = {}
        for key, value in metrics.items():
            if value is None:
                quantized[key] = None
                continue

            # Similarity-based metrics [0,1] → uint16
            if any(
                keyword in key.lower()
                for keyword in ["similarity", "coherence", "density", "robustness"]
            ):
                if "coherence" in key.lower():
                    # Internal coherence: typically [0, 1], but invert for storage
                    quantized[key] = np.uint16(
                        np.clip((1.0 - float(value)) * 65535, 0, 65535)
                    )
                else:
                    quantized[key] = np.uint16(np.clip(float(value) * 65535, 0, 65535))

            # Silhouette score [-1, 1] → uint16 with offset
            elif "silhouette" in key.lower():
                # Map [-1, 1] to [0, 65535]
                normalized = (float(value) + 1.0) / 2.0
                quantized[key] = np.uint16(np.clip(normalized * 65535, 0, 65535))

            # Distance metrics → float32
            elif "distance" in key.lower():
                if "normalized" in key.lower():
                    # Normalized distances [0,1] → uint16
                    quantized[key] = np.uint16(np.clip(float(value) * 65535, 0, 65535))
                else:
                    quantized[key] = np.float32(value)

            else:
                # Default: keep as float32
                quantized[key] = np.float32(value)

        return quantized

    def _dequantize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float | None]:
        """Dequantize metrics back to their original ranges.

        Reverses the quantization process to restore original metric values.

        Args:
            metrics: Dictionary of quantized metric values.

        Returns:
            Dictionary with dequantized float values.
        """
        if not self.quantization_enabled:
            return {k: float(v) if v is not None else None for k, v in metrics.items()}

        dequantized: Dict[str, float | None] = {}
        for key, value in metrics.items():
            if value is None:
                dequantized[key] = None
                continue

            # Handle quantized uint16 values
            if (isinstance(value, (np.integer, int))) and value <= 65535:
                if "coherence" in key.lower():
                    # Internal coherence was inverted: restore original
                    dequantized[key] = 1.0 - (float(value) / 65535.0)
                elif "silhouette" in key.lower():
                    # Silhouette: map [0, 65535] back to [-1, 1]
                    normalized = float(value) / 65535.0
                    dequantized[key] = (normalized * 2.0) - 1.0
                elif any(
                    keyword in key.lower()
                    for keyword in ["similarity", "density", "robustness", "normalized"]
                ):
                    # Standard [0,1] metrics
                    dequantized[key] = float(value) / 65535.0
                else:
                    dequantized[key] = float(value)
            else:
                dequantized[key] = float(value)

        return dequantized

    def get_db_modification_time(self) -> float:
        """Get the last modification time of the database file.

        Returns:
            Unix timestamp of the last modification.
        """
        return os.path.getmtime(self.db_path)

    def _apply_intelligent_quantization(self, obj: Any) -> Any:
        """Apply intelligent quantization based on data type and content.

        Optimizes storage by quantizing data based on its type and expected
        value ranges.

        Args:
            obj: Object to quantize.

        Returns:
            Quantized object with reduced precision where appropriate.
        """
        if not self.quantization_enabled:
            return obj

        if isinstance(obj, dict):
            quantized = {}
            for key, value in obj.items():
                if key in [
                    "similarities",
                    "cosine_similarity",
                    "dot_product",
                ] and isinstance(value, (np.ndarray, list)):
                    # Similarity metrics - quantifier seulement si dans [0,1]
                    if isinstance(value, list):
                        value = np.array(value)
                    if (
                        value.dtype.kind == "f"
                        and 0 <= value.min()
                        and value.max()
                        <= 1.01  # Petite tolérance pour les erreurs de floating point
                    ):
                        quantized[key] = (value * 65535).astype(np.uint16)
                    else:
                        # Garder en float32 pour dot_product et autres métriques non-normalisées
                        quantized[key] = value.astype(np.float32)
                elif key == "scatter_plot_data" and isinstance(value, dict):
                    # 2D coordinates → float16
                    scatter_quantized = {}
                    for scatter_key, scatter_value in value.items():
                        if scatter_key in ["x", "y"] and isinstance(
                            scatter_value, (list, np.ndarray)
                        ):
                            scatter_quantized[scatter_key] = np.array(
                                scatter_value, dtype=np.float16
                            )
                        elif scatter_key == "similarities" and isinstance(
                            scatter_value, (list, np.ndarray)
                        ):
                            # Similarities in scatter plot data
                            if isinstance(scatter_value, list):
                                scatter_value = np.array(scatter_value)
                            if (
                                scatter_value.dtype.kind == "f"
                                and 0 <= scatter_value.min()
                                and scatter_value.max() <= 1
                            ):
                                scatter_quantized[scatter_key] = (
                                    scatter_value * 65535
                                ).astype(np.uint16)
                            else:
                                scatter_quantized[scatter_key] = scatter_value
                        else:
                            scatter_quantized[scatter_key] = scatter_value
                    quantized[key] = scatter_quantized
                elif key == "metrics" and isinstance(value, dict):
                    # Apply metric-specific quantization only if enabled
                    if self.quantize_metrics:
                        quantized[key] = self._quantize_metrics(value)
                    else:
                        quantized[key] = {k: np.float32(v) if isinstance(v, float) else v 
                                         for k, v in value.items()}
                else:
                    quantized[key] = self._apply_intelligent_quantization(value)
            return quantized
        elif isinstance(obj, np.ndarray):
            if obj.dtype == np.float64:
                # Downcast float64 → float32
                return obj.astype(np.float32)
            elif obj.dtype.kind == "f" and obj.ndim > 1:
                # Embeddings: if normalized [-1,1], use float16
                if -1.1 <= obj.min() and obj.max() <= 1.1:
                    return obj.astype(np.float16)
        elif isinstance(obj, list):
            return [self._apply_intelligent_quantization(item) for item in obj]

        return obj

    def _restore_quantized_data(self, obj: Any) -> Any:
        """Restore quantized data to original format.

        Reverses the quantization process to restore full precision.

        Args:
            obj: Quantized object to restore.

        Returns:
            Object with restored full-precision values.
        """
        if not self.quantization_enabled:
            return obj

        if isinstance(obj, dict):
            restored = {}
            for key, value in obj.items():
                if key in [
                    "similarities",
                    "cosine_similarity",
                    "dot_product",
                ] and isinstance(value, np.ndarray):
                    # Restore uint16 → float32 seulement si c'était quantifié
                    if value.dtype == np.uint16:
                        restored[key] = value.astype(np.float32) / 65535.0
                    else:
                        # Était déjà en float32 (dot_product, etc.)
                        restored[key] = value.astype(np.float32)
                elif key == "scatter_plot_data" and isinstance(value, dict):
                    scatter_restored = {}
                    for scatter_key, scatter_value in value.items():
                        if scatter_key == "similarities" and isinstance(
                            scatter_value, np.ndarray
                        ):
                            if scatter_value.dtype == np.uint16:
                                scatter_restored[scatter_key] = (
                                    scatter_value.astype(np.float32) / 65535.0
                                )
                            else:
                                scatter_restored[scatter_key] = scatter_value
                        else:
                            scatter_restored[scatter_key] = scatter_value
                    restored[key] = scatter_restored
                elif key == "metrics" and isinstance(value, dict):
                    # Apply metric-specific dequantization
                    restored[key] = self._dequantize_metrics(value)
                else:
                    restored[key] = self._restore_quantized_data(value)
            return restored
        elif isinstance(obj, list):
            return [self._restore_quantized_data(item) for item in obj]

        return obj

    def _numpy_default(self, obj: Any) -> Any:
        """Custom encoder for numpy data types for msgpack.

        Handles numpy arrays by compressing them with zlib.

        Args:
            obj: Object to encode.

        Returns:
            Encoded representation suitable for msgpack.

        Raises:
            TypeError: If the object is not serializable.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            # Use maximum compression for cache storage (embeddings don't change often)
            compression_level = 9 if hasattr(self, "_cache_storage") else 6
            return {
                "__ndarray__": True,
                "dtype": obj.dtype.str,
                "shape": obj.shape,
                "data": zlib.compress(obj.tobytes(), level=compression_level),
            }
        raise TypeError(f"Object of type {obj.__class__.__name__} is not serializable")

    def _decode_numpy(self, obj: Any) -> Any:
        """Custom decoder for numpy data types for msgpack.

        Decompresses and reconstructs numpy arrays from msgpack format.

        Args:
            obj: Object to decode.

        Returns:
            Decoded object, with numpy arrays reconstructed.
        """
        if isinstance(obj, dict) and "__ndarray__" in obj:
            data = zlib.decompress(obj["data"])
            return np.frombuffer(data, dtype=np.dtype(obj["dtype"])).reshape(
                obj["shape"]
            )
        return obj
