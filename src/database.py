import os
import sqlite3
import zlib
from typing import Any, Dict, List, Optional, Tuple

import msgpack
import numpy as np


class EmbeddingDatabase:
    """Database manager for storing embedding results."""

    def __init__(
        self,
        db_path: str = "data/heatmaps/ForzaEmbed.db",
        quantize: bool = False,
        intelligent_quantization: bool = False,
    ):
        self.db_path = db_path
        self.quantize = quantize
        self.intelligent_quantization = intelligent_quantization
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()

    def _apply_intelligent_quantization(self, obj: Any) -> Any:
        """Apply intelligent quantization based on data type and content."""
        if not self.intelligent_quantization:
            return obj

        if isinstance(obj, dict):
            quantized = {}
            for key, value in obj.items():
                if key in [
                    "similarities",
                    "cosine_similarity",
                    "dot_product",
                ] and isinstance(value, np.ndarray):
                    # Similarity metrics [0,1] → uint16
                    if (
                        value.dtype.kind == "f"
                        and 0 <= value.min()
                        and value.max() <= 1
                    ):
                        quantized[key] = (value * 65535).astype(np.uint16)
                    else:
                        quantized[key] = self._apply_intelligent_quantization(value)
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
                        else:
                            scatter_quantized[scatter_key] = scatter_value
                    quantized[key] = scatter_quantized
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
        """Restore quantized data to original format."""
        if not self.intelligent_quantization:
            return obj

        if isinstance(obj, dict):
            restored = {}
            for key, value in obj.items():
                if key in [
                    "similarities",
                    "cosine_similarity",
                    "dot_product",
                ] and isinstance(value, np.ndarray):
                    # Restore uint16 → float32
                    if value.dtype == np.uint16:
                        restored[key] = value.astype(np.float32) / 65535.0
                    else:
                        restored[key] = self._restore_quantized_data(value)
                elif key == "scatter_plot_data" and isinstance(value, dict):
                    # Keep float16 for coordinates (sufficient precision)
                    restored[key] = {k: v for k, v in value.items()}
                else:
                    restored[key] = self._restore_quantized_data(value)
            return restored
        elif isinstance(obj, list):
            return [self._restore_quantized_data(item) for item in obj]

        return obj

    def _numpy_default(self, obj):
        """Custom encoder for numpy data types for msgpack, with compression."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            # Apply legacy quantization if enabled
            if self.quantize and obj.dtype.kind == "f":
                obj = obj.astype(np.float16)

            # Use maximum compression for cache storage (embeddings don't change often)
            compression_level = 9 if hasattr(self, "_cache_storage") else 6
            return {
                "__ndarray__": True,
                "dtype": obj.dtype.str,
                "shape": obj.shape,
                "data": zlib.compress(obj.tobytes(), level=compression_level),
            }
        raise TypeError(f"Object of type {obj.__class__.__name__} is not serializable")

    def _decode_numpy(self, obj):
        """Custom decoder for numpy data types for msgpack."""
        if isinstance(obj, dict) and "__ndarray__" in obj:
            data = zlib.decompress(obj["data"])
            return np.frombuffer(data, dtype=np.dtype(obj["dtype"])).reshape(
                obj["shape"]
            )
        return obj

    def get_db_modification_time(self) -> float:
        """Returns the last modification time of the database file."""
        return os.path.getmtime(self.db_path)

    def init_database(self):
        """Initializes the database with the necessary tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    base_model_name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    chunk_size INTEGER NOT NULL,
                    chunk_overlap INTEGER NOT NULL,
                    theme_name TEXT NOT NULL,
                    chunking_strategy TEXT NOT NULL,
                    similarity_metric TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Evaluation metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    cohesion REAL,
                    separation REAL,
                    discriminant_score REAL,
                    silhouette REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_name) REFERENCES models (name)
                )
            """)

            # Generated files table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generated_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_name) REFERENCES models (name)
                )
            """)

            # Global comparison charts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS global_charts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chart_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Table for storing detailed processing results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    results_blob BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(model_name, file_id)
                )
            """)

            # Check if embedding_cache needs migration
            cursor.execute("PRAGMA table_info(embedding_cache)")
            columns = [row[1] for row in cursor.fetchall()]

            if "cache_key" in columns or not columns:
                # Migration needed: drop old table and recreate with composite primary key
                if columns:
                    print("Migrating embedding_cache table structure...")
                cursor.execute("DROP TABLE IF EXISTS embedding_cache")

            # Table for caching embeddings - Optimized with composite primary key
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    model_name TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (model_name, text_hash)
                )
            """)

            # Nouvelle table pour stocker les coordonnées t-SNE
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tsne_coordinates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tsne_key TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    coordinates BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(tsne_key, file_id)
                )
            """)

            conn.commit()

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
    ):
        """Adds a model to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO models (name, base_model_name, type, chunk_size, chunk_overlap, theme_name, chunking_strategy, similarity_metric)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    name,
                    base_model_name,
                    model_type,
                    chunk_size,
                    chunk_overlap,
                    theme_name,
                    chunking_strategy,
                    similarity_metric,
                ),
            )
            conn.commit()

    def add_evaluation_metrics(self, model_name: str, metrics: Dict[str, float]):
        """Adds or updates evaluation metrics for a model."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Delete existing metrics for this model to avoid duplicates
            cursor.execute(
                "DELETE FROM evaluation_metrics WHERE model_name = ?", (model_name,)
            )
            # Insert the new metrics
            cursor.execute(
                """
                INSERT INTO evaluation_metrics 
                (model_name, cohesion, separation, discriminant_score, 
                 silhouette)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    model_name,
                    metrics.get("cohesion"),
                    metrics.get("separation"),
                    metrics.get("discriminant_score"),
                    metrics.get("silhouette"),
                ),
            )
            conn.commit()

    def add_generated_file(self, model_name: str, file_type: str, file_path: str):
        """Adds a generated file to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO generated_files (model_name, file_type, file_path)
                VALUES (?, ?, ?)
            """,
                (model_name, file_type, file_path),
            )
            conn.commit()

    def add_global_chart(self, chart_type: str, file_path: str):
        """Adds or updates a global chart in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Delete existing chart of this type
            cursor.execute(
                "DELETE FROM global_charts WHERE chart_type = ?", (chart_type,)
            )
            # Insert the new chart path
            cursor.execute(
                """
                INSERT INTO global_charts (chart_type, file_path)
                VALUES (?, ?)
            """,
                (chart_type, file_path),
            )
            conn.commit()

    def model_exists(self, name: str) -> bool:
        """Checks if a model with the specified run name already exists."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM models WHERE name = ?", (name,))
            return cursor.fetchone() is not None

    def save_processing_result(
        self, model_name: str, file_id: str, results: Dict[str, Any]
    ):
        """Saves the detailed processing result for a file and a model."""
        # Apply intelligent quantization before serialization
        quantized_results = self._apply_intelligent_quantization(results)
        results_blob = msgpack.packb(
            quantized_results, default=self._numpy_default, use_bin_type=True
        )

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO processing_results (model_name, file_id, results_blob)
                VALUES (?, ?, ?)
            """,
                (model_name, file_id, results_blob),
            )
            conn.commit()

    def save_processing_results_batch(
        self, results_batch: List[Tuple[str, str, Dict[str, Any]]]
    ):
        """Saves a batch of processing results in a single transaction."""
        items_to_insert = []
        for model_name, file_id, results in results_batch:
            # Apply intelligent quantization before serialization
            quantized_results = self._apply_intelligent_quantization(results)
            results_blob = msgpack.packb(
                quantized_results, default=self._numpy_default, use_bin_type=True
            )
            items_to_insert.append((model_name, file_id, results_blob))

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR REPLACE INTO processing_results (model_name, file_id, results_blob)
                VALUES (?, ?, ?)
            """,
                items_to_insert,
            )
            conn.commit()

    def get_processed_files(self, model_name: str) -> List[str]:
        """Retrieves the list of file_ids that have been processed for a model."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT file_id FROM processing_results WHERE model_name = ?",
                (model_name,),
            )
            return [row[0] for row in cursor.fetchall()]

    def get_model_info(self, run_name: str) -> Optional[Dict[str, Any]]:
        """Retrieves information about a model by its run name."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT base_model_name, type, chunk_size, chunk_overlap, theme_name, chunking_strategy
                FROM models
                WHERE name = ?
            """,
                (run_name,),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "model_name": row[0],
                    "type": row[1],
                    "chunk_size": row[2],
                    "chunk_overlap": row[3],
                    "theme_name": row[4],
                    "chunking_strategy": row[5],
                }
            return None

    def get_all_processing_results(self) -> Dict[str, Any]:
        """
        Retrieves all processing results, including aggregated embeddings and labels,
        grouped by model run. This is a comprehensive fetch for all reporting needs.
        """
        all_results: Dict[str, Dict[str, Any]] = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT model_name, file_id, results_blob FROM processing_results"
            )

            for row in cursor.fetchall():
                model_name, file_id, results_blob = row
                # Unpack the blob and restore any quantized data
                results = msgpack.unpackb(
                    results_blob, object_hook=self._decode_numpy, raw=False
                )
                results = self._restore_quantized_data(results)

                if model_name not in all_results:
                    all_results[model_name] = {
                        "files": {},
                        "embeddings": [],
                        "labels": [],
                    }

                # Store file-specific results
                all_results[model_name]["files"][file_id] = results

                # Aggregate embeddings and labels at the model level
                if "embeddings" in results and results["embeddings"] is not None:
                    all_results[model_name]["embeddings"].extend(results["embeddings"])
                if "labels" in results and results["labels"] is not None:
                    all_results[model_name]["labels"].extend(results["labels"])

        # Convert lists to numpy arrays for processing
        for model_name in all_results:
            if all_results[model_name]["embeddings"]:
                all_results[model_name]["embeddings"] = np.array(
                    all_results[model_name]["embeddings"]
                )
            else:
                # Ensure it's an empty array with correct shape if no embeddings found
                all_results[model_name]["embeddings"] = np.array([]).reshape(0, 0)

        return all_results

    def get_all_models(self) -> List[Dict[str, Any]]:
        """Retrieves all models with their metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT m.name, m.base_model_name, m.type, m.chunk_size, m.chunk_overlap, m.theme_name, m.chunking_strategy,
                       e.cohesion, e.separation, e.discriminant_score,
                       e.silhouette
                FROM models m
                LEFT JOIN evaluation_metrics e ON m.name = e.model_name
                ORDER BY m.name
            """
            )

            models = []
            for row in cursor.fetchall():
                models.append(
                    {
                        "name": row[0],
                        "base_model_name": row[1],
                        "type": row[2],
                        "chunk_size": row[3],
                        "chunk_overlap": row[4],
                        "theme_name": row[5],
                        "chunking_strategy": row[6],
                        "cohesion": row[7],
                        "separation": row[8],
                        "discriminant_score": row[9],
                        "silhouette": row[10],
                    }
                )

            return models

    def get_model_files(self, model_name: str) -> List[Dict[str, str]]:
        """Retrieves all generated files for a model."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT file_type, file_path FROM generated_files
                WHERE model_name = ?
                ORDER BY file_type
            """,
                (model_name,),
            )

            return [{"type": row[0], "path": row[1]} for row in cursor.fetchall()]

    def get_global_charts(self) -> List[Dict[str, str]]:
        """Retrieves all global charts."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT chart_type, file_path FROM global_charts
                ORDER BY chart_type
            """)

            return [{"type": row[0], "path": row[1]} for row in cursor.fetchall()]

    def vacuum_database(self):
        """Vacuums the database to reclaim space."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("VACUUM")

    def clear_database(self):
        """Clears all tables in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM generated_files")
            cursor.execute("DELETE FROM evaluation_metrics")
            cursor.execute("DELETE FROM global_charts")
            cursor.execute("DELETE FROM processing_results")
            cursor.execute("DELETE FROM models")
            cursor.execute("DELETE FROM embedding_cache")
            cursor.execute("DELETE FROM tsne_coordinates")
            conn.commit()

    def get_all_run_names(self) -> list[str]:
        """Récupère tous les run_names existants."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT name FROM models")
            return [row[0] for row in cursor.fetchall()]

    def get_processed_files_with_similarities(self, run_name: str) -> list[str]:
        """
        Récupère la liste des fichiers qui ont été traités avec des similarités calculées
        pour un run donné.
        """
        processed_files = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT file_id, results_blob FROM processing_results WHERE model_name = ?",
                (run_name,),
            )
            for file_id, results_blob in cursor.fetchall():
                results = msgpack.unpackb(
                    results_blob, object_hook=self._decode_numpy, raw=False
                )
                if "similarities" in results and results["similarities"] is not None:
                    processed_files.append(file_id)
        return processed_files

    def get_embeddings_by_hashes(
        self, base_model_name: str, text_hashes: List[str]
    ) -> Dict[str, np.ndarray]:
        """Retrieves embeddings from the cache by base model and text hashes."""
        if not text_hashes:
            return {}

        placeholders = ",".join("?" for _ in text_hashes)
        query_params = [base_model_name] + text_hashes

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT text_hash, vector FROM embedding_cache WHERE model_name = ? AND text_hash IN ({placeholders})",
                query_params,
            )

            embeddings = {}
            for text_hash, vector_blob in cursor.fetchall():
                embeddings[text_hash] = msgpack.unpackb(
                    vector_blob, object_hook=self._decode_numpy, raw=False
                )
            return embeddings

    def save_embeddings_batch(
        self, base_model_name: str, embeddings: Dict[str, np.ndarray]
    ):
        """Saves a batch of embeddings to the cache using the base model name."""
        if not embeddings:
            return

        # Flag for maximum compression in cache
        self._cache_storage = True

        items_to_insert = []
        for text_hash, vector in embeddings.items():
            # Apply intelligent quantization to embeddings for cache storage
            if self.intelligent_quantization and vector.dtype.kind == "f":
                # Most embeddings are normalized [-1,1] → float16 is sufficient
                if -1.1 <= vector.min() and vector.max() <= 1.1:
                    vector = vector.astype(np.float16)
                elif vector.dtype == np.float64:
                    # Downcast float64 → float32
                    vector = vector.astype(np.float32)

            # Apply legacy quantization if enabled (for backward compatibility)
            elif self.quantize and vector.dtype.kind == "f":
                vector = vector.astype(np.float16)

            vector_blob = msgpack.packb(
                vector, default=self._numpy_default, use_bin_type=True
            )
            dimension = len(vector) if len(vector.shape) == 1 else vector.shape[1]
            items_to_insert.append((base_model_name, text_hash, vector_blob, dimension))

        delattr(self, "_cache_storage")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                "INSERT OR IGNORE INTO embedding_cache (model_name, text_hash, vector, dimension) VALUES (?, ?, ?, ?)",
                items_to_insert,
            )
            conn.commit()

    def clear_embedding_cache(self):
        """Clears all embeddings from the cache."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM embedding_cache")
            conn.commit()

    def save_tsne_coordinates(
        self, tsne_key: str, file_id: str, coordinates: Dict[str, List[float]]
    ):
        """Sauvegarde les coordonnées t-SNE pour une combinaison donnée."""
        coordinates_blob = msgpack.packb(coordinates, use_bin_type=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO tsne_coordinates (tsne_key, file_id, coordinates)
                VALUES (?, ?, ?)
                """,
                (tsne_key, file_id, coordinates_blob),
            )
            conn.commit()

    def get_tsne_coordinates(
        self, tsne_key: str, file_id: str
    ) -> Optional[Dict[str, List[float]]]:
        """Récupère les coordonnées t-SNE pour une combinaison donnée."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT coordinates FROM tsne_coordinates WHERE tsne_key = ? AND file_id = ?",
                (tsne_key, file_id),
            )
            row = cursor.fetchone()
            if row:
                return msgpack.unpackb(row[0], raw=False)
            return None

    def clear_tsne_cache(self):
        """Vide le cache des coordonnées t-SNE."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM tsne_coordinates")
            conn.commit()

    def get_run_details(self, run_name: str) -> Optional[Dict[str, Any]]:
        """Retrieves detailed information for a specific run."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM models WHERE name = ?", (run_name,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_processing_results_for_run(
        self, model_name: str
    ) -> Dict[str, Dict[str, Any]]:
        """Get all processing results for a specific run with proper dequantization."""
        results = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT file_id, results_blob FROM processing_results WHERE model_name = ?",
                (model_name,),
            )
            for file_id, results_blob in cursor.fetchall():
                data = msgpack.unpackb(
                    results_blob, object_hook=self._decode_numpy, raw=False
                )
                # Restore quantized data
                results[file_id] = self._restore_quantized_data(data)
        return results

    def _dequantize_similarities(self, similarities):
        """Ensure similarities are properly dequantized to [0,1] range"""
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
    ):
        """Updates the metrics for a specific file in a model run."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # First, retrieve the existing blob
            cursor.execute(
                "SELECT results_blob FROM processing_results WHERE model_name = ? AND file_id = ?",
                (model_name, file_id),
            )
            row = cursor.fetchone()
            if not row:
                return  # Or handle error

            # Deserialize, update, and re-serialize
            results_data = msgpack.unpackb(
                row[0], object_hook=self._decode_numpy, raw=False
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
            cursor.execute(
                "UPDATE processing_results SET results_blob = ? WHERE model_name = ? AND file_id = ?",
                (updated_blob, model_name, file_id),
            )
            conn.commit()

    def _quantize_metrics(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Quantize metrics intelligently based on their expected ranges."""
        if not self.intelligent_quantization:
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
        """Dequantize metrics back to their original ranges."""
        if not self.intelligent_quantization:
            return {k: float(v) if v is not None else None for k, v in metrics.items()}

        dequantized: Dict[str, float | None] = {}
        for key, value in metrics.items():
            if value is None:
                dequantized[key] = None
                continue

            # Handle quantized uint16 values
            if (
                isinstance(value, np.integer) or isinstance(value, int)
            ) and value <= 65535:
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
