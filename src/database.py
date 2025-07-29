import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils import to_python_type


class EmbeddingDatabase:
    """Gestionnaire de base de données pour stocker les résultats d'embedding."""

    def __init__(self, db_path: str = "data/heatmaps/embedding_results.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()

    def init_database(self):
        """Initialise la base de données avec les tables nécessaires."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Table des modèles
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Table des métriques d'évaluation
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

            # Table des fichiers générés
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

            # Table des graphiques de comparaison globaux
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS global_charts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chart_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Table pour stocker les résultats de traitement détaillés
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
    ):
        """Ajoute un modèle à la base de données."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO models (name, base_model_name, type, chunk_size, chunk_overlap, theme_name, chunking_strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    name,
                    base_model_name,
                    model_type,
                    chunk_size,
                    chunk_overlap,
                    theme_name,
                    chunking_strategy,
                ),
            )
            conn.commit()

    def add_evaluation_metrics(self, model_name: str, metrics: Dict[str, float]):
        """Ajoute les métriques d'évaluation pour un modèle."""
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
        """Ajoute un fichier généré à la base de données."""
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
        """Ajoute un graphique global à la base de données."""
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

    def save_processing_result(
        self, model_name: str, file_id: str, results: Dict[str, Any]
    ):
        """Sauvegarde le résultat de traitement détaillé pour un fichier et un modèle."""
        # Convertir les arrays numpy en listes pour la sérialisation JSON
        if "embeddings_data" in results and "embeddings" in results["embeddings_data"]:
            if isinstance(results["embeddings_data"].get("embeddings"), np.ndarray):
                results["embeddings_data"]["embeddings"] = results["embeddings_data"][
                    "embeddings"
                ].tolist()
            if isinstance(results["embeddings_data"].get("labels"), np.ndarray):
                results["embeddings_data"]["labels"] = results["embeddings_data"][
                    "labels"
                ].tolist()

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

    def get_model_info(self, run_name: str) -> Optional[Dict[str, Any]]:
        """Récupère les informations d'un modèle par son nom de run."""
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
        """Récupère tous les résultats de traitement depuis la base de données."""
        all_results = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT model_name, file_id, results_json FROM processing_results"
            )

            for row in cursor.fetchall():
                model_name, file_id, results_json = row
                results = json.loads(results_json)

                # Reconvertir les listes en arrays numpy si nécessaire
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
        """Récupère tous les modèles avec leurs métriques."""
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
        """Récupère tous les fichiers générés pour un modèle."""
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
        """Récupère tous les graphiques globaux."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT chart_type, file_path FROM global_charts
                ORDER BY chart_type
            """)

            return [{"type": row[0], "path": row[1]} for row in cursor.fetchall()]

    def clear_database(self):
        """Vide toutes les tables de la base de données."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM generated_files")
            cursor.execute("DELETE FROM evaluation_metrics")
            cursor.execute("DELETE FROM global_charts")
            cursor.execute("DELETE FROM processing_results")
            cursor.execute("DELETE FROM models")
            conn.commit()
