import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils import to_python_type


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
                    results_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(model_name, file_id)
                )
            """)

            # Cache for phrase embeddings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings_cache (
                    model_name TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    PRIMARY KEY (model_name, text_hash)
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
        """Adds evaluation metrics for a model."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO evaluation_metrics 
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
        """Adds a global chart to the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO global_charts (chart_type, file_path)
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
        # Convert all numpy objects to native Python types for JSON serialization
        results_json = json.dumps(to_python_type(results))

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO processing_results (model_name, file_id, results_json)
                VALUES (?, ?, ?)
            """,
                (model_name, file_id, results_json),
            )
            conn.commit()

    def save_processing_results_batch(
        self, results_batch: List[Tuple[str, str, Dict[str, Any]]]
    ):
        """Saves a batch of processing results in a single transaction."""
        items_to_insert = []
        for model_name, file_id, results in results_batch:
            # Convert all numpy objects to native Python types for JSON serialization
            results_json = json.dumps(to_python_type(results))
            items_to_insert.append((model_name, file_id, results_json))

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR REPLACE INTO processing_results (model_name, file_id, results_json)
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

    def get_cached_embeddings(
        self, model_name: str, text_hashes: List[str]
    ) -> Dict[str, List[float]]:
        """Retrieves cached embeddings for a list of text hashes."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in text_hashes)
            query = f"SELECT text_hash, embedding_json FROM embeddings_cache WHERE model_name = ? AND text_hash IN ({placeholders})"
            cursor.execute(query, [model_name] + text_hashes)

            cached_embeddings = {}
            for row in cursor.fetchall():
                text_hash, embedding_json = row
                cached_embeddings[text_hash] = json.loads(embedding_json)
            return cached_embeddings

    def cache_embeddings(self, model_name: str, embeddings_map: Dict[str, List[float]]):
        """Caches multiple embeddings in a single transaction."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            items_to_insert = [
                (model_name, text_hash, json.dumps(to_python_type(embedding)))
                for text_hash, embedding in embeddings_map.items()
            ]
            cursor.executemany(
                "INSERT OR IGNORE INTO embeddings_cache (model_name, text_hash, embedding_json) VALUES (?, ?, ?)",
                items_to_insert,
            )
            conn.commit()

    def get_all_processing_results(self) -> Dict[str, Any]:
        """Retrieivs all processing results from the database."""
        all_results = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT model_name, file_id, results_json FROM processing_results"
            )

            for row in cursor.fetchall():
                model_name, file_id, results_json = row
                results = json.loads(results_json)

                # Convert lists back to numpy arrays if necessary
                if (
                    "embeddings_data" in results
                    and "embeddings" in results["embeddings_data"]
                ):
                    results["embeddings_data"]["embeddings"] = np.array(
                        results["embeddings_data"]["embeddings"]
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

    def clear_database(self):
        """Clears all tables in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM generated_files")
            cursor.execute("DELETE FROM evaluation_metrics")
            cursor.execute("DELETE FROM global_charts")
            cursor.execute("DELETE FROM processing_results")
            cursor.execute("DELETE FROM embeddings_cache")
            cursor.execute("DELETE FROM models")
            conn.commit()
