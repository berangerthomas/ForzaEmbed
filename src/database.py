import os
import sqlite3
import zlib
from typing import Any, Dict, List, Optional, Tuple

import msgpack
import numpy as np


def numpy_default(obj):
    """Custom encoder for numpy data types for msgpack, with compression."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        # Using zlib for a good balance of speed and compression
        return {
            "__ndarray__": True,
            "dtype": obj.dtype.str,
            "shape": obj.shape,
            "data": zlib.compress(obj.tobytes()),
        }
    raise TypeError(f"Object of type {obj.__class__.__name__} is not serializable")


def decode_numpy(obj):
    """Custom decoder for numpy data types for msgpack."""
    if isinstance(obj, dict) and "__ndarray__" in obj:
        data = zlib.decompress(obj["data"])
        return np.frombuffer(data, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
    return obj


class EmbeddingDatabase:
    """Database manager for storing embedding results."""

    def __init__(self, db_path: str = "data/heatmaps/ForzaEmbed.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()

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
                    calinski_harabasz REAL,
                    davies_bouldin REAL,
                    processing_time REAL,
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

            if "cache_key" not in columns:
                # Migration needed: drop old table and recreate
                print("Migrating embedding_cache table structure...")
                cursor.execute("DROP TABLE IF EXISTS embedding_cache")

            # Table for caching embeddings - MODIFIED to include model_name
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    cache_key TEXT PRIMARY KEY NOT NULL,
                    model_name TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                 silhouette, calinski_harabasz, davies_bouldin, processing_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    model_name,
                    metrics.get("cohesion"),
                    metrics.get("separation"),
                    metrics.get("discriminant_score"),
                    metrics.get("silhouette"),
                    metrics.get("calinski_harabasz"),
                    metrics.get("davies_bouldin"),
                    metrics.get("processing_time"),
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
        results_blob = msgpack.packb(results, default=numpy_default, use_bin_type=True)

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
            results_blob = msgpack.packb(
                results, default=numpy_default, use_bin_type=True
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
        """Retrieivs all processing results from the database."""
        all_results = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT model_name, file_id, results_blob FROM processing_results"
            )

            for row in cursor.fetchall():
                model_name, file_id, results_blob = row
                results = msgpack.unpackb(
                    results_blob, object_hook=decode_numpy, raw=False
                )

                if model_name not in all_results:
                    all_results[model_name] = {"files": {}}
                all_results[model_name]["files"][file_id] = results
        return all_results

    def get_all_models(self) -> List[Dict[str, Any]]:
        """Retrieves all models with their metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT m.name, m.base_model_name, m.type, m.chunk_size, m.chunk_overlap, m.theme_name, m.chunking_strategy,
                       e.cohesion, e.separation, e.discriminant_score,
                       e.silhouette, e.calinski_harabasz, e.davies_bouldin,
                       e.processing_time
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
                        "calinski_harabasz": row[11],
                        "davies_bouldin": row[12],
                        "processing_time": row[13],
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
                    results_blob, object_hook=decode_numpy, raw=False
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

        # Use the base_model_name for the cache key to ensure one entry per model
        cache_keys = [f"{base_model_name}:{h}" for h in text_hashes]
        placeholders = ",".join("?" for _ in cache_keys)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT cache_key, vector FROM embedding_cache WHERE cache_key IN ({placeholders})",
                cache_keys,
            )

            embeddings = {}
            for cache_key, vector_blob in cursor.fetchall():
                # The text hash is after the first colon
                text_hash = cache_key.split(":", 1)[1]
                embeddings[text_hash] = msgpack.unpackb(
                    vector_blob, object_hook=decode_numpy, raw=False
                )
            return embeddings

    def save_embeddings_batch(
        self, base_model_name: str, embeddings: Dict[str, np.ndarray]
    ):
        """Saves a batch of embeddings to the cache using the base model name."""
        if not embeddings:
            return

        items_to_insert = []
        for text_hash, vector in embeddings.items():
            # Use the base_model_name for the cache key
            cache_key = f"{base_model_name}:{text_hash}"
            vector_blob = msgpack.packb(
                vector, default=numpy_default, use_bin_type=True
            )
            dimension = len(vector) if len(vector.shape) == 1 else vector.shape[1]
            items_to_insert.append(
                (cache_key, base_model_name, text_hash, vector_blob, dimension)
            )

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                "INSERT OR IGNORE INTO embedding_cache (cache_key, model_name, text_hash, vector, dimension) VALUES (?, ?, ?, ?, ?)",
                items_to_insert,
            )
            conn.commit()

    def clear_embedding_cache(self):
        """Clears all embeddings from the cache."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM embedding_cache")
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
        """Retrieves all processing results for a specific model run."""
        results = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT file_id, results_blob FROM processing_results WHERE model_name = ?",
                (model_name,),
            )
            for file_id, results_blob in cursor.fetchall():
                results[file_id] = msgpack.unpackb(
                    results_blob, object_hook=decode_numpy, raw=False
                )
        return results

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
            results_data = msgpack.unpackb(row[0], object_hook=decode_numpy, raw=False)
            results_data["metrics"].update(metrics)
            updated_blob = msgpack.packb(
                results_data, default=numpy_default, use_bin_type=True
            )

            # Update the blob in the database
            cursor.execute(
                "UPDATE processing_results SET results_blob = ? WHERE model_name = ? AND file_id = ?",
                (updated_blob, model_name, file_id),
            )
            conn.commit()
