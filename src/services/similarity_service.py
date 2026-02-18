"""Similarity calculation service for ForzaEmbed.

This module provides the SimilarityService class that handles various
similarity and distance metric calculations between embeddings. It supports
cosine, dot product, euclidean, manhattan, and chebyshev metrics.

Example:
    Calculate similarity between theme and phrase embeddings::

        from src.services.similarity_service import SimilarityService

        similarities = SimilarityService.calculate_similarity(
            embed_themes, embed_phrases, "cosine"
        )
        validated = SimilarityService.validate_similarities(similarities, "cosine")
"""

import logging

import numpy as np
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
    pairwise_distances,
)


class SimilarityService:
    """Handle similarity calculations and validation.

    Provides static methods for computing various similarity metrics
    between embedding matrices and validating/normalizing the results.
    """

    @staticmethod
    def calculate_similarity(
        embed_themes: np.ndarray, embed_phrases: np.ndarray, metric: str
    ) -> np.ndarray:
        """Calculate similarity between theme embeddings and phrase embeddings.

        Args:
            embed_themes: Theme embeddings array of shape (n_themes, n_dims).
            embed_phrases: Phrase embeddings array of shape (n_phrases, n_dims).
            metric: The similarity metric to use. One of 'cosine', 'dot_product',
                'euclidean', 'manhattan', or 'chebyshev'.

        Returns:
            Similarity matrix of shape (n_themes, n_phrases).

        Raises:
            ValueError: If an unknown similarity metric is specified.
        """
        similarity_functions: dict[str, callable] = {
            "cosine": cosine_similarity,
            "dot_product": lambda themes, phrases: themes @ phrases.T,
            "euclidean": lambda themes, phrases: 1 / (1 + euclidean_distances(themes, phrases)),
            "manhattan": lambda themes, phrases: 1 / (1 + manhattan_distances(themes, phrases)),
            "chebyshev": lambda themes, phrases: 1 / (1 + pairwise_distances(themes, phrases, metric="chebyshev")),
        }

        if metric in similarity_functions:
            return similarity_functions[metric](embed_themes, embed_phrases)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

    @staticmethod
    def validate_similarities(
        similarities: np.ndarray, metric: str
    ) -> np.ndarray:
        """Validate and clean similarities based on the metric used.

        Handles NaN and infinite values, then normalizes the similarity
        values to an appropriate range based on the metric type.

        Args:
            similarities: Raw similarity matrix to validate.
            metric: The similarity metric that was used. One of 'cosine',
                'dot_product', 'euclidean', 'manhattan', or 'chebyshev'.

        Returns:
            Cleaned and normalized similarity matrix with values in [0, 1].
        """
        # Replace NaN and inf with appropriate values
        similarities = np.nan_to_num(similarities, nan=0.0, posinf=1.0, neginf=0.0)

        # Validation based on metric
        if metric == "cosine":
            # Cosine similarity is in [-1, 1]. Normalize to [0, 1].
            if similarities.min() < -1.1 or similarities.max() > 1.1:
                logging.warning(
                    f"Cosine similarities out of expected range [-1,1]: min={similarities.min()}, max={similarities.max()}"
                )
            similarities = (similarities + 1.0) / 2.0
        elif metric == "dot_product":
            # Normalize dot product to [0, 1] using min-max normalization
            # This preserves relative differences while ensuring valid range for display
            min_val = similarities.min()
            max_val = similarities.max()
            if max_val > min_val:
                similarities = (similarities - min_val) / (max_val - min_val)
            else:
                # All values are the same, set to 0.5 (neutral)
                similarities = np.full_like(similarities, 0.5)
        elif metric in ["euclidean", "manhattan", "chebyshev"]:
            # Distance metrics converted to similarity: [0, +inf) → similarity in [0, 1]
            # Values are already converted by 1/(1+distance), so should be in [0, 1]
            if similarities.min() < 0 or similarities.max() > 1.1:
                logging.warning(
                    f"{metric} similarities out of expected range [0,1]: min={similarities.min()}, max={similarities.max()}"
                )
            # Remove clipping to preserve information
            # similarities = np.clip(similarities, 0.0, 1.0)

        return similarities
