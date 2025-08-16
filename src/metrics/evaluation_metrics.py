"""Evaluation metrics for text embeddings based on similarity scores."""

from typing import Dict, List

import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from .silhouette_decomposition import enhanced_silhouette_analysis


def coherence_score(ref_embeddings: np.ndarray, doc_embeddings: np.ndarray) -> float:
    """Calculates the Internal Coherence Score (ICS).

    This metric assesses the stability and predictability of the similarity
    measurement system. A reliable system should produce consistent similarity
    scores for similar items, rather than wildly fluctuating, random scores.
    The score is calculated as the average ratio of variance to the mean of
    similarity scores for each reference embedding against all document
    embeddings.

    Interpretation:
        - ICS < 0.1: Excellent. The system is highly coherent and stable.
        - 0.1 <= ICS < 0.5: Good to Acceptable. The system shows reasonable
          coherence.
        - ICS >= 0.5: Poor. The system is unstable, and scores are erratic.
        A lower score indicates higher coherence.

    Formula:
        ICS = (1/n) * Σ [var(sim(e_i, E_doc)) / mean(sim(e_i, E_doc))]
        Where:
        - n: Number of reference embeddings.
        - e_i: A single reference embedding.
        - E_doc: The set of all document embeddings.
        - sim(a, B): Similarity scores between embedding 'a' and all
          embeddings in set 'B'.
        - var: Variance of the similarity scores.
        - mean: Mean of the similarity scores.

    Args:
        ref_embeddings (np.ndarray): Embeddings for the reference themes.
        doc_embeddings (np.ndarray): Embeddings for the document chunks.

    Returns:
        float: The Internal Coherence Score. A lower value is better.
    """
    coherence_scores: List[float] = []
    for ref_emb in ref_embeddings:
        similarities = cosine_similarity(
            ref_emb.reshape(1, -1), doc_embeddings
        ).flatten()
        if similarities.size > 0:
            mean_sim = np.mean(similarities)
            var_sim = np.var(similarities)
            if mean_sim > 1e-8:  # Avoid division by zero
                coherence_scores.append(float(var_sim / mean_sim))
    return float(np.mean(coherence_scores)) if coherence_scores else 0.0


def local_density_index(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
    metric: str = "cosine",
    p: int = 2,
) -> float:
    """Calculates the Local Density Index (LDI).

    This metric assesses whether the embedding space has a meaningful
    structure by checking if an embedding's nearest neighbors belong to the
    same theme. It measures the proportion of k-nearest neighbors that share
    the same label as the point in question.

    Interpretation:
        - LDI = 1.0: Perfect. All k-nearest neighbors for every point belong
          to the same theme.
        - LDI > 0.8: Good. The embedding space has a strong thematic
          structure.
        - LDI < 0.5: Poor. The neighbors are distributed almost randomly,
          indicating a weak structure.
        A higher score is better.

    Formula:
        LDI = Σ |N_k(e_i) ∩ Theme(e_i)| / (k * n)
        Where:
        - n: Total number of embeddings.
        - k: Number of nearest neighbors to consider.
        - N_k(e_i): The set of k-nearest neighbors of embedding e_i.
        - Theme(e_i): The set of embeddings belonging to the same theme as e_i.
        - |...|: The number of elements in the set.

    Args:
        embeddings (np.ndarray): The embeddings of the text chunks.
        labels (np.ndarray): The theme label for each chunk.
        k (int): The number of nearest neighbors to consider.
        p (int): The power parameter for the Minkowski distance.

    Returns:
        float: The Local Density Index score, between 0 and 1.
    """
    if len(embeddings) < k + 1:
        return 0.0
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric, p=p).fit(embeddings)
    indices = nbrs.kneighbors(embeddings, return_distance=False)
    same_theme_count = 0
    for i, neighbors in enumerate(indices):
        current_theme = labels[i]
        # Exclude the point itself (it's always the first neighbor)
        neighbor_themes = labels[neighbors[1:]]
        same_theme_count += np.sum(neighbor_themes == current_theme)
    return float(same_theme_count / (k * len(embeddings)))


def robustness_score(
    ref_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    noise_level: float = 0.01,
) -> float:
    """Calculates the Robustness Score (RS) by adding Gaussian noise.

    This metric tests the stability of the similarity system by introducing a
    small amount of random noise to the reference embeddings and measuring the
    change in similarity scores. A robust system should not be significantly
    affected by such minor perturbations.

    Interpretation:
        - RS > 0.95: Very Robust. The system is highly stable.
        - 0.80 <= RS <= 0.95: Acceptable. The system shows minor variations.
        - RS < 0.80: Fragile. The system is sensitive to noise and unstable.
        A higher score (closer to 1.0) is better.

    Formula:
        RS = 1 - |S(E_ref, E_doc) - S(E_ref + ε, E_doc)| / S(E_ref, E_doc)
        Where:
        - S(A, B): The mean similarity score between embedding sets A and B.
        - E_ref: The set of reference embeddings.
        - E_doc: The set of document embeddings.
        - ε: A small Gaussian noise vector, ε ~ N(0, σ²I).

    Args:
        ref_embeddings (np.ndarray): Embeddings for reference themes.
        doc_embeddings (np.ndarray): Embeddings for document chunks.
        noise_level (float): The standard deviation (σ) of the Gaussian
          noise to add.

    Returns:
        float: The Robustness Score, typically between 0 and 1.
    """
    if ref_embeddings.size == 0 or doc_embeddings.size == 0:
        return 0.0
    base_similarity = np.mean(cosine_similarity(ref_embeddings, doc_embeddings))
    if abs(base_similarity) < 1e-8:
        return 0.0
    noise = np.random.normal(0, noise_level, ref_embeddings.shape)
    perturbed_embeddings = ref_embeddings + noise
    perturbed_similarity = np.mean(
        cosine_similarity(perturbed_embeddings, doc_embeddings)
    )
    rs = 1 - abs(base_similarity - perturbed_similarity) / abs(base_similarity)
    return float(rs)


def calculate_silhouette_metrics(
    embeddings: np.ndarray, labels: np.ndarray, metric: str = "cosine"
) -> Dict[str, float]:
    """Calculates silhouette-based clustering metrics with normalized components.

    This function decomposes the silhouette score into its constituent parts:
    intra-cluster distance (cohesion) and inter-cluster distance (separation),
    providing normalized versions for better interpretability.

    Args:
        embeddings (np.ndarray): The embeddings of the text chunks.
        labels (np.ndarray): The theme label for each chunk.
        metric (str): Distance metric to use for calculations.

    Returns:
        Dict[str, float]: Dictionary containing:
            - intra_cluster_distance_normalized: Normalized intra-cluster quality (0-1, higher is better)
            - inter_cluster_distance_normalized: Normalized inter-cluster separation (0-1, higher is better)
            - silhouette_score: Standard silhouette score (-1 to 1, higher is better)
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
    a_values = []  # intra-cluster distances
    b_values = []  # inter-cluster distances

    unique_labels = np.unique(labels)

    for i in range(n_samples):
        current_label = labels[i]

        # a(i): Average intra-cluster distance
        same_cluster_mask = (labels == current_label) & (np.arange(n_samples) != i)
        if np.sum(same_cluster_mask) > 0:
            a_i = np.mean(distance_matrix[i][same_cluster_mask])
        else:
            a_i = 0.0
        a_values.append(a_i)

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
        b_values.append(b_i)

    a_values = np.array(a_values)
    b_values = np.array(b_values)

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
    intra_normalized = 1 - (np.mean(a_values) / max_possible_distance)

    # Inter-cluster separation: average_distance / max_distance
    # Higher values indicate better separation (clusters farther apart)
    inter_normalized = np.mean(b_values) / max_possible_distance

    return {
        "intra_cluster_distance_normalized": float(max(0.0, float(intra_normalized))),
        "inter_cluster_distance_normalized": float(inter_normalized),
        "silhouette_score": float(silhouette_computed),
    }


def calculate_all_metrics(
    ref_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    doc_labels: np.ndarray,
) -> Dict[str, float]:
    """Calculates and combines all evaluation metrics.

    This function serves as a primary entry point to compute a comprehensive
    set of metrics for evaluating the quality of document embeddings against a
    set of reference themes.

    Args:
        ref_embeddings (np.ndarray): Embeddings for reference themes.
        doc_embeddings (np.ndarray): Embeddings for document chunks.
        doc_labels (np.ndarray): Theme labels for each document chunk.

    Returns:
        Dict[str, float]: A dictionary containing all calculated metrics.
    """
    all_metrics: Dict[str, float] = {}

    # Custom metrics
    all_metrics["internal_coherence_score"] = coherence_score(
        ref_embeddings, doc_embeddings
    )
    all_metrics["robustness_score"] = robustness_score(ref_embeddings, doc_embeddings)
    all_metrics["local_density_index"] = local_density_index(doc_embeddings, doc_labels)

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
