
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from ..utils.database import EmbeddingDatabase

class VisualizationService:
    """
    Handles visualization tasks like t-SNE coordinate generation.
    """
    def __init__(self, db: EmbeddingDatabase):
        self.db = db

    def get_or_create_tsne_data(
        self,
        embeddings: np.ndarray,
        tsne_key: str,
        file_id: str,
        similarities: np.ndarray,
        threshold: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Calcule ou récupère les coordonnées t-SNE pour une combinaison donnée.
        """
        if embeddings.shape[0] <= 1:
            return None

        # Vérifier si les coordonnées t-SNE existent déjà
        cached_tsne = self.db.get_tsne_coordinates(tsne_key, file_id)

        if cached_tsne is not None:
            # Utiliser les coordonnées existantes mais recalculer les labels selon les nouvelles similarités
            similarity_scores = similarities.max(axis=0)
            scatter_labels = [
                "Above Threshold" if s >= threshold else "Below Threshold"
                for s in similarity_scores
            ]

            # S'assurer que toutes les données sont des types Python natifs
            tsne_data = {
                "x": self._safe_convert_to_python_types(cached_tsne["x"]),
                "y": self._safe_convert_to_python_types(cached_tsne["y"]),
                "labels": scatter_labels,
                "similarities": self._safe_convert_to_python_types(similarity_scores),
                "title": f"t-SNE Visualization for {file_id}",
                "threshold": float(threshold),
            }
            return tsne_data

        # Calculer de nouvelles coordonnées t-SNE
        try:
            from sklearn.manifold import TSNE

            tsne = TSNE(
                n_components=2,
                perplexity=min(30, embeddings.shape[0] - 1),
                random_state=42,
                max_iter=300,
            )
            tsne_results = tsne.fit_transform(embeddings)

            # Sauvegarder les coordonnées pour réutilisation
            tsne_coords = {
                "x": tsne_results[:, 0].astype(float).tolist(),
                "y": tsne_results[:, 1].astype(float).tolist(),
            }
            self.db.save_tsne_coordinates(tsne_key, file_id, tsne_coords)

            # Calculer les labels selon les similarités actuelles
            similarity_scores = similarities.max(axis=0)
            scatter_labels = [
                "Above Threshold" if s >= threshold else "Below Threshold"
                for s in similarity_scores
            ]

            return {
                "x": tsne_coords["x"],
                "y": tsne_coords["y"],
                "labels": scatter_labels,
                "similarities": self._safe_convert_to_python_types(similarity_scores),
                "title": f"t-SNE Visualization for {file_id}",
                "threshold": float(threshold),
            }

        except Exception as e:
            logging.error(f"Error during t-SNE calculation for {file_id}: {e}")
            return None

    def _safe_convert_to_python_types(self, data: Any) -> Any:
        """
        Convertit récursivement tous les types NumPy en types Python natifs.
        """
        if isinstance(data, np.ndarray):
            return data.astype(float).tolist()
        elif isinstance(data, (np.floating, float)):
            return float(data)
        elif isinstance(data, (np.integer, int)):
            return int(data)
        elif isinstance(data, dict):
            return {
                key: self._safe_convert_to_python_types(value)
                for key, value in data.items()
            }
        elif isinstance(data, (list, tuple)):
            return [self._safe_convert_to_python_types(item) for item in data]
        else:
            return data
