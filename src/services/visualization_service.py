"""Visualization service for ForzaEmbed.

This module provides the VisualizationService class that handles
dimensionality reduction (t-SNE, UMAP, PCA) and caching for embedding
visualizations.
"""

import logging
from typing import Any, Dict

import numpy as np

from ..utils.database import EmbeddingDatabase


class VisualizationService:
    """Handle visualization tasks like UMAP, PCA and t-SNE coordinate generation.

    Manages the computation and caching of projection coordinates for
    embedding visualizations.

    Attributes:
        db: The embedding database for caching coordinates.
    """

    def __init__(self, db: EmbeddingDatabase) -> None:
        """Initialize the VisualizationService.

        Args:
            db: The embedding database for caching.
        """
        self.db = db

    def get_or_create_projections(
        self,
        embeddings: np.ndarray,
        base_key: str,
        file_id: str,
        similarities: np.ndarray,
    ) -> Dict[str, Any] | None:
        """Compute or retrieve projection coordinates (UMAP, t-SNE, PCA).

        Checks the database cache for existing coordinates using method-specific keys.

        Args:
            embeddings: Embedding matrix of shape (n_samples, n_dims).
            base_key: Base cache key for the computation.
            file_id: Identifier for the file being visualized.
            similarities: Similarity matrix for determining labels.
            threshold: Similarity threshold for labeling points.

        Returns:
            Dictionary containing projection data for umap, tsne, and pca.
            Returns None if embeddings have <= 1 sample or on error.
        """
        if embeddings.shape[0] <= 1:
            return None

        similarity_scores = similarities.max(axis=0)
        safe_similarities = self._safe_convert_to_python_types(similarity_scores)
        
        methods = ["umap", "tsne", "pca"]
        results = {}

        for method in methods:
            cache_key = f"{base_key}_{method}"
            cached_data = self.db.get_projection_coordinates(cache_key, file_id)

            if cached_data is not None:
                coords = cached_data
                results[method] = {
                    "x": self._safe_convert_to_python_types(cached_data["x"]),
                    "y": self._safe_convert_to_python_types(cached_data["y"]),
                    "similarities": safe_similarities,
                    "title": f"Visualization for {file_id}",
                }
                for k, v in cached_data.items():
                    if k not in ["x", "y", "labels", "threshold"]:
                        results[method][k] = v
                continue

            # Compute new coordinates
            try:
                if method == "umap":
                    import umap
                    n_neighbors = min(15, embeddings.shape[0] - 1)
                    n_neighbors = max(2, n_neighbors)
                    reducer = umap.UMAP(
                        n_neighbors=n_neighbors,
                        min_dist=0.1,
                        n_components=2,
                        random_state=42,
                        metric='cosine'
                    )
                    projections = reducer.fit_transform(embeddings)
                    coords = {
                        "x": projections[:, 0].astype(float).tolist(),
                        "y": projections[:, 1].astype(float).tolist(),
                        "n_neighbors": n_neighbors,
                        "min_dist": 0.1,
                        "metric": 'cosine'
                    }

                elif method == "pca":
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2, random_state=42)
                    projections = pca.fit_transform(embeddings)
                    variance_ratio = pca.explained_variance_ratio_
                    coords = {
                        "x": projections[:, 0].astype(float).tolist(),
                        "y": projections[:, 1].astype(float).tolist(),
                        "explained_variance_1": float(variance_ratio[0]),
                        "explained_variance_2": float(variance_ratio[1])
                    }

                elif method == "tsne":
                    from sklearn.manifold import TSNE
                    perplexity_val = min(30, embeddings.shape[0] - 1)
                    perplexity_val = max(1, perplexity_val)
                    tsne = TSNE(
                        n_components=2,
                        perplexity=perplexity_val,
                        random_state=42,
                        max_iter=1000,
                        init="pca",
                        learning_rate="auto",
                    )
                    projections = tsne.fit_transform(embeddings)
                    coords = {
                        "x": projections[:, 0].astype(float).tolist(),
                        "y": projections[:, 1].astype(float).tolist(),
                        "kl_divergence": float(getattr(tsne, 'kl_divergence_', 0.0)),
                        "n_iter": int(getattr(tsne, 'n_iter_', 0)),
                        "perplexity": perplexity_val,
                        "max_iter": 1000,
                        "init": "pca",
                        "learning_rate": "auto",
                    }

                self.db.save_projection_coordinates(cache_key, file_id, coords)
                
                method_result = {
                    "x": coords["x"],
                    "y": coords["y"],
                    "similarities": safe_similarities,
                    "title": f"Visualization for {file_id}",
                }
                for k, v in coords.items():
                    if k not in ["x", "y"]:
                        method_result[k] = v
                
                results[method] = method_result

            except Exception as e:
                logging.error(f"Error during {method} calculation for {file_id}: {e}")
                
        return results

    def _safe_convert_to_python_types(self, data: Any) -> Any:
        """Recursively convert all NumPy types to native Python types."""
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
