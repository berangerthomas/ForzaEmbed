"""Silhouette score decomposition into intra-cluster and inter-cluster components.

This module provides functions for decomposing the silhouette score into
its constituent parts: intra-cluster cohesion (a(i)) and inter-cluster
separation (b(i)). This decomposition helps understand clustering quality
in more detail than the aggregate silhouette score alone.

Example:
    Perform enhanced silhouette analysis::

        from src.metrics.silhouette_decomposition import enhanced_silhouette_analysis

        analysis = enhanced_silhouette_analysis(embeddings, labels)
        print(f"Global metrics: {analysis['global_metrics']}")
        print(f"Per-cluster: {analysis['cluster_analysis']}")
"""

from typing import Any

import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_samples, silhouette_score


def decompose_silhouette_score(
    embeddings: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Decompose the silhouette score into its a(i) and b(i) components.

    The silhouette score s(i) = (b(i) - a(i)) / max(a(i), b(i)) where:
    - a(i) = average intra-cluster distance (cohesion) - LOWER = BETTER
    - b(i) = average distance to nearest cluster - HIGHER = BETTER

    Args:
        embeddings: Embedding matrix of shape (n_samples, n_features).
        labels: Cluster labels of shape (n_samples,).

    Returns:
        Dictionary containing:
            - mean_intra_cluster_distance: Average a(i) across samples.
            - mean_inter_cluster_distance: Average b(i) across samples.
            - silhouette_score: Aggregate silhouette score.
            - intra_cluster_quality: Normalized cohesion (0-1, higher = better).
            - inter_cluster_separation: Normalized separation (0-1, higher = better).
    """
    n_samples = len(embeddings)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Strict validation: need at least 2 clusters AND enough samples
    if n_clusters < 2 or n_samples <= n_clusters:
        return {
            "mean_intra_cluster_distance": 0.0,
            "mean_inter_cluster_distance": 0.0,
            "silhouette_score": -1.0,
            "intra_cluster_quality": 0.0,
            "inter_cluster_separation": 0.0,
        }

    # Always use 'cosine' for silhouette analysis
    clustering_metric = "cosine"

    # Compute distance matrix
    distance_matrix = pairwise_distances(embeddings, metric=clustering_metric)

    a_values: list[float] = []  # intra-cluster cohesion
    b_values: list[float] = []  # inter-cluster separation

    for i in range(n_samples):
        current_label = labels[i]

        # a(i): Average intra-cluster distance
        same_cluster_mask = (labels == current_label) & (np.arange(n_samples) != i)
        if np.sum(same_cluster_mask) > 0:
            a_i = float(np.mean(distance_matrix[i][same_cluster_mask]))
        else:
            a_i = 0.0
        a_values.append(a_i)

        # b(i): Average distance to nearest cluster
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

    # Compute silhouette score for verification
    silhouette_computed = silhouette_score(embeddings, labels, metric=clustering_metric)

    # Normalization for interpretable metrics (0-1)
    max_possible_distance = float(np.max(distance_matrix))

    # Intra-cluster quality: 1 - (mean_distance / max_distance)
    # Closer to 1 = better cohesion
    intra_quality = (
        1 - (np.mean(a_values_arr) / max_possible_distance)
        if max_possible_distance > 0
        else 0
    )

    # Inter-cluster separation: mean_distance / max_distance
    # Closer to 1 = better separation
    inter_separation = (
        np.mean(b_values_arr) / max_possible_distance if max_possible_distance > 0 else 0
    )

    return {
        "mean_intra_cluster_distance": float(np.mean(a_values_arr)),
        "mean_inter_cluster_distance": float(np.mean(b_values_arr)),
        "silhouette_score": float(silhouette_computed),
        "intra_cluster_quality": float(intra_quality),
        "inter_cluster_separation": float(inter_separation),
    }


def analyze_silhouette_by_cluster(
    embeddings: np.ndarray, labels: np.ndarray
) -> dict[int, dict[str, float]]:
    """Perform detailed silhouette score analysis per cluster.

    Args:
        embeddings: Embedding matrix of shape (n_samples, n_features).
        labels: Cluster labels of shape (n_samples,).

    Returns:
        Dictionary mapping cluster label to its silhouette statistics:
            - mean_silhouette: Average silhouette score for the cluster.
            - std_silhouette: Standard deviation of silhouette scores.
            - min_silhouette: Minimum silhouette score in cluster.
            - max_silhouette: Maximum silhouette score in cluster.
            - size: Number of samples in the cluster.
            - proportion_positive: Fraction of samples with positive score.
        Returns empty dict if fewer than 2 clusters or insufficient samples.
    """
    n_samples = len(embeddings)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Strict validation: same condition as decompose_silhouette_score
    if n_clusters < 2 or n_samples <= n_clusters:
        return {}

    sample_scores: np.ndarray = np.array(
        silhouette_samples(embeddings, labels, metric="cosine")
    )

    cluster_analysis: dict[int, dict[str, float]] = {}

    for label in unique_labels:
        cluster_mask = labels == label
        cluster_silhouettes = sample_scores[cluster_mask]

        if cluster_silhouettes.size == 0:
            continue

        cluster_analysis[int(label)] = {
            "mean_silhouette": float(np.mean(cluster_silhouettes)),
            "std_silhouette": float(np.std(cluster_silhouettes)),
            "min_silhouette": float(np.min(cluster_silhouettes)),
            "max_silhouette": float(np.max(cluster_silhouettes)),
            "size": int(np.sum(cluster_mask)),
            "proportion_positive": float(np.mean(cluster_silhouettes > 0)),
        }

    return cluster_analysis


def enhanced_silhouette_analysis(
    embeddings: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    """Perform complete clustering analysis with silhouette decomposition.

    Combines global silhouette decomposition with per-cluster analysis to
    provide a comprehensive view of clustering quality.

    Note:
        Always uses 'cosine' as the distance metric for clustering analysis,
        regardless of the similarity metric used for embedding evaluation.

    Args:
        embeddings: Embedding matrix of shape (n_samples, n_features).
        labels: Cluster labels of shape (n_samples,).

    Returns:
        Dictionary containing:
            - global_metrics: Results from decompose_silhouette_score().
            - cluster_analysis: Results from analyze_silhouette_by_cluster().
    """
    global_decomp = decompose_silhouette_score(embeddings, labels)
    cluster_analysis = analyze_silhouette_by_cluster(embeddings, labels)

    return {"global_metrics": global_decomp, "cluster_analysis": cluster_analysis}
