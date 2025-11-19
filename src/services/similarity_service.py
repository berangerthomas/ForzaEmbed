
import logging
import numpy as np
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
    pairwise_distances,
)

class SimilarityService:
    """
    Handles similarity calculations and validation.
    """
    
    @staticmethod
    def calculate_similarity(
        embed_themes: np.ndarray, embed_phrases: np.ndarray, metric: str
    ) -> np.ndarray:
        """Calculate similarity between theme embeddings and phrase embeddings."""
        
        similarity_functions = {
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
        """
        Valide et nettoie les similarités selon la métrique utilisée.
        """
        # Remplacer NaN et inf par des valeurs appropriées
        similarities = np.nan_to_num(similarities, nan=0.0, posinf=1.0, neginf=0.0)

        # Validation selon la métrique
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
            # Distance metrics convertis en similarité: [0, +inf) → similarity dans [0, 1]
            # Les valeurs sont déjà converties par 1/(1+distance), donc devraient être dans [0, 1]
            if similarities.min() < 0 or similarities.max() > 1.1:
                logging.warning(
                    f"{metric} similarities out of expected range [0,1]: min={similarities.min()}, max={similarities.max()}"
                )
            # Remove clipping to preserve information
            # similarities = np.clip(similarities, 0.0, 1.0)

        return similarities
