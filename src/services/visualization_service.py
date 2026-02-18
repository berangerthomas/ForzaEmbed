"""Visualization service for ForzaEmbed.

This module provides the VisualizationService class that handles
t-SNE coordinate generation and caching for embedding visualizations.

Example:
    Generate t-SNE visualization data::

        from src.services.visualization_service import VisualizationService

        service = VisualizationService(db)
        tsne_data = service.get_or_create_tsne_data(
            embeddings, "key", "file_id", similarities, 0.5
        )
"""

import logging
from typing import Any

import numpy as np

from ..utils.database import EmbeddingDatabase


class VisualizationService:
    """Handle visualization tasks like t-SNE coordinate generation.

    Manages the computation and caching of t-SNE coordinates for
    embedding visualizations.

    Attributes:
        db: The embedding database for caching t-SNE coordinates.
    """

    def __init__(self, db: EmbeddingDatabase) -> None:
        """Initialize the VisualizationService.

        Args:
            db: The embedding database for caching.
        """
        self.db = db

    def get_or_create_tsne_data(
        self,
        embeddings: np.ndarray,
        tsne_key: str,
        file_id: str,
        similarities: np.ndarray,
        threshold: float,
    ) -> dict[str, Any] | None:
        """Compute or retrieve t-SNE coordinates for a given combination.

        Checks the database cache for existing t-SNE coordinates. If not
        found, computes new coordinates using sklearn's TSNE implementation.

        Args:
            embeddings: Embedding matrix of shape (n_samples, n_dims).
            tsne_key: Cache key for the t-SNE computation.
            file_id: Identifier for the file being visualized.
            similarities: Similarity matrix for determining labels.
            threshold: Similarity threshold for labeling points.

        Returns:
            Dictionary containing t-SNE visualization data with keys:
                - 'x': List of x-coordinates.
                - 'y': List of y-coordinates.
                - 'labels': List of threshold-based labels.
                - 'similarities': List of similarity scores.
                - 'title': Visualization title.
                - 'threshold': The threshold value used.
            Returns None if embeddings have <= 1 sample or on error.
        """
        if embeddings.shape[0] <= 1:
            return None

        # Check if t-SNE coordinates already exist
        cached_tsne = self.db.get_tsne_coordinates(tsne_key, file_id)

        if cached_tsne is not None:
            # Use existing coordinates but recalculate labels based on new similarities
            similarity_scores = similarities.max(axis=0)
            scatter_labels = [
                "Above Threshold" if s >= threshold else "Below Threshold"
                for s in similarity_scores
            ]

            # Ensure all data is native Python types
            tsne_data = {
                "x": self._safe_convert_to_python_types(cached_tsne["x"]),
                "y": self._safe_convert_to_python_types(cached_tsne["y"]),
                "labels": scatter_labels,
                "similarities": self._safe_convert_to_python_types(similarity_scores),
                "title": f"t-SNE Visualization for {file_id}",
                "threshold": float(threshold),
            }
            return tsne_data

        # Compute new t-SNE coordinates
        try:
            from sklearn.manifold import TSNE

            tsne = TSNE(
                n_components=2,
                perplexity=min(30, embeddings.shape[0] - 1),
                random_state=42,
                max_iter=300,
            )
            tsne_results = tsne.fit_transform(embeddings)

            # Save coordinates for reuse
            tsne_coords = {
                "x": tsne_results[:, 0].astype(float).tolist(),
                "y": tsne_results[:, 1].astype(float).tolist(),
            }
            self.db.save_tsne_coordinates(tsne_key, file_id, tsne_coords)

            # Calculate labels based on current similarities
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
        """Recursively convert all NumPy types to native Python types.

        Args:
            data: Data to convert, can be ndarray, scalar, dict, list, or other.

        Returns:
            Data with all NumPy types converted to native Python equivalents.
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
