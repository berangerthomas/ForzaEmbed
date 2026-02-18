"""Evaluation metrics for text embeddings based on similarity scores.

This module provides functions for calculating clustering quality metrics
on embedding spaces, particularly silhouette-based metrics that decompose
into intra-cluster cohesion and inter-cluster separation.

Example:
    Calculate metrics for document embeddings::

        from src.metrics.evaluation_metrics import calculate_all_metrics

        metrics = calculate_all_metrics(ref_embeddings, doc_embeddings, doc_labels)
        print(f"Silhouette score: {metrics['silhouette_score']}")
"""

import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_score

from .silhouette_decomposition import enhanced_silhouette_analysis


def calculate_silhouette_metrics(
    embeddings: np.ndarray, labels: np.ndarray, metric: str = "cosine"
) -> dict[str, float]:
    """Calculate silhouette-based clustering metrics with normalized components.

    This function decomposes the silhouette score into its constituent parts:
    intra-cluster distance (cohesion) and inter-cluster distance (separation),
    providing normalized versions for better interpretability.

    Args:
        embeddings: The embeddings of the text chunks, shape (n_samples, n_dims).
        labels: The theme label for each chunk, shape (n_samples,).
        metric: Distance metric to use for calculations. Defaults to 'cosine'.

    Returns:
        Dictionary containing:
            - intra_cluster_distance_normalized: Normalized intra-cluster
              quality (0-1, higher is better).
            - inter_cluster_distance_normalized: Normalized inter-cluster
              separation (0-1, higher is better).
            - silhouette_score: Standard silhouette score (-1 to 1, higher
              is better).
    """
    if len(np.unique(labels)) < 2:
        return {
            "intra_cluster_distance_normalized": 0.0,
            "inter_cluster_distance_normalized": 0.0,
            "silhouette_score": -1.0,
        }

    # Calculate distance matrix
    distance_matrix = pairwise_distances(embeddings, metric=metric)

    n_samples = len(embeddings)
    a_values: list[float] = []  # intra-cluster distances
    b_values: list[float] = []  # inter-cluster distances

    unique_labels = np.unique(labels)

    for i in range(n_samples):
        current_label = labels[i]

        # a(i): Average intra-cluster distance
        same_cluster_mask = (labels == current_label) & (np.arange(n_samples) != i)
        if np.sum(same_cluster_mask) > 0:
            a_i = np.mean(distance_matrix[i][same_cluster_mask])
        else:
            a_i = 0.0
        a_values.append(float(a_i))

        # b(i): Average distance to nearest different cluster
        b_i = np.inf
        for other_label in unique_labels:
            if other_label != current_label:
                other_cluster_mask = labels == other_label
                if np.sum(other_cluster_mask) > 0:
                    mean_dist_to_other = np.mean(distance_matrix[i][other_cluster_mask])
                    b_i = min(b_i, mean_dist_to_other)

        if b_i == np.inf:
            b_i = 0.0
        b_values.append(float(b_i))

    a_values_arr = np.array(a_values)
    b_values_arr = np.array(b_values)

    # Calculate silhouette score using sklearn for robustness
    try:
        silhouette_computed = silhouette_score(embeddings, labels, metric=metric)
    except (ValueError, IndexError):
        silhouette_computed = -1.0

    # Normalize metrics for interpretability
    max_possible_distance = (
        np.max(distance_matrix) if np.max(distance_matrix) > 0 else 1.0
    )

    # Intra-cluster quality: 1 - (average_distance / max_distance)
    # Higher values indicate better cohesion (points closer within clusters)
    intra_normalized = 1 - (np.mean(a_values_arr) / max_possible_distance)

    # Inter-cluster separation: average_distance / max_distance
    # Higher values indicate better separation (clusters farther apart)
    inter_normalized = np.mean(b_values_arr) / max_possible_distance

    return {
        "intra_cluster_distance_normalized": float(max(0.0, float(intra_normalized))),
        "inter_cluster_distance_normalized": float(inter_normalized),
        "silhouette_score": float(silhouette_computed),
    }


def calculate_all_metrics(
    ref_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    doc_labels: np.ndarray,
) -> dict[str, float]:
    """Calculate minimal essential evaluation metrics.

    This function computes only the core metrics needed for clustering
    evaluation: silhouette score and its decomposition (intra/inter
    cluster distances).

    Args:
        ref_embeddings: Embeddings for reference themes, shape (n_themes, n_dims).
        doc_embeddings: Embeddings for document chunks, shape (n_chunks, n_dims).
        doc_labels: Theme labels for each document chunk, shape (n_chunks,).

    Returns:
        Dictionary containing silhouette-based metrics with keys:
            - silhouette_score
            - intra_cluster_distance_normalized
            - inter_cluster_distance_normalized
    """
    all_metrics: dict[str, float] = {}

    # Enhanced silhouette analysis for clustering quality
    if len(np.unique(doc_labels)) > 1:
        silhouette_analysis = enhanced_silhouette_analysis(doc_embeddings, doc_labels)
        global_metrics = silhouette_analysis["global_metrics"]
        all_metrics.update(
            {
                "silhouette_score": global_metrics["silhouette_score"],
                "intra_cluster_distance_normalized": global_metrics[
                    "intra_cluster_quality"
                ],
                "inter_cluster_distance_normalized": global_metrics[
                    "inter_cluster_separation"
                ],
            }
        )
    else:
        # Provide default values if silhouette score cannot be computed
        all_metrics.update(
            {
                "silhouette_score": -1.0,
                "intra_cluster_distance_normalized": 0.0,
                "inter_cluster_distance_normalized": 0.0,
            }
        )

    return all_metrics
