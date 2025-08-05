"""Evaluation metrics for text embeddings based on similarity scores."""

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


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


def angular_stability(ref_embeddings: np.ndarray, doc_embeddings: np.ndarray) -> float:
    """Calculates the Angular Stability (AS).

    This metric measures whether the reference theme embeddings "point" in the
    same general direction as the document's overall embedding representation.
    It computes the average angle between each reference embedding and the
    centroid of the document embeddings.

    Interpretation (angles in radians, with degree equivalents):
        - 0 to 0.52 (~0-30°): Excellent alignment. Themes are very
          relevant to the document.
        - 0.52 to 1.05 (~30-60°): Good alignment. Themes are relevant.
        - 1.05 to 1.57 (~60-90°): Weak alignment. Themes are not very
          relevant.
        - > 1.57 (>90°): Opposition. Themes might be contradictory to the
          document's content.
        A lower angle indicates better alignment.

    Formula:
        AS = (1/|E_ref|) * Σ ∠(e_i, centroid(E_doc))
        Where:
        - |E_ref|: The number of reference embeddings.
        - e_i: A single reference embedding.
        - centroid(E_doc): The mean vector (centroid) of the document
          embeddings.
        - ∠(a, b): The angle between vectors 'a' and 'b'.

    Args:
        ref_embeddings (np.ndarray): Embeddings for the reference themes.
        doc_embeddings (np.ndarray): Embeddings for the document chunks.

    Returns:
        float: The average angle in radians. A lower value is better.
    """
    if doc_embeddings.size == 0 or ref_embeddings.size == 0:
        return float("inf")
    doc_centroid = np.mean(doc_embeddings, axis=0).reshape(1, -1)
    similarities = cosine_similarity(ref_embeddings, doc_centroid).flatten()
    # Clip to avoid domain errors with arccos for values slightly out of [-1, 1]
    clipped_similarities = np.clip(similarities, -1.0, 1.0)
    angles = np.arccos(clipped_similarities)
    return float(np.mean(angles))


def local_density_index(
    embeddings: np.ndarray, labels: np.ndarray, k: int = 5
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

    Returns:
        float: The Local Density Index score, between 0 and 1.
    """
    if len(embeddings) < k + 1:
        return 0.0
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(embeddings)
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


def calculate_cohesion_separation(
    embeddings: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    """Calculates cohesion, separation, and a discriminant score.

    - Cohesion: Measures the average similarity between embeddings within the
      same theme. A higher score indicates that chunks of the same theme are
      semantically close.
    - Separation: Measures the average similarity between embeddings from
      different themes. A lower score indicates that different themes are
      well-distinguished.
    - Discriminant Score: The ratio of cohesion to separation. A higher score
      indicates better overall clustering quality.

    Args:
        embeddings (np.ndarray): The embeddings of the text chunks.
        labels (np.ndarray): The theme label for each chunk.

    Returns:
        Dict[str, float]: A dictionary containing cohesion, separation, and
        a combined discriminant score.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {
            "cohesion": 0.0,
            "separation": 0.0,
            "discriminant_score": 0.0,
        }

    # Cohesion: Average similarity within each theme
    cohesion_scores: List[float] = []
    for label in unique_labels:
        theme_embeddings = embeddings[labels == label]
        if len(theme_embeddings) > 1:
            similarity_matrix = cosine_similarity(theme_embeddings)
            # Use upper triangle to avoid diagonal and duplicates
            indices = np.triu_indices_from(similarity_matrix, k=1)
            if similarity_matrix[indices].size > 0:
                cohesion_scores.append(float(np.mean(similarity_matrix[indices])))

    avg_cohesion = np.mean(cohesion_scores) if cohesion_scores else 0.0

    # Separation: Average similarity between different themes
    separation_scores: List[float] = []
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            label1 = unique_labels[i]
            label2 = unique_labels[j]
            embeddings1 = embeddings[labels == label1]
            embeddings2 = embeddings[labels == label2]
            if embeddings1.size > 0 and embeddings2.size > 0:
                similarity_matrix = cosine_similarity(embeddings1, embeddings2)
                separation_scores.append(float(np.mean(similarity_matrix)))

    avg_separation = np.mean(separation_scores) if separation_scores else 0.0

    # Discriminant Score
    discriminant_score = avg_cohesion / avg_separation if avg_separation > 1e-8 else 0.0

    return {
        "cohesion": float(avg_cohesion),
        "separation": float(avg_separation),
        "discriminant_score": float(discriminant_score),
    }


def calculate_clustering_metrics(
    embeddings: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    """Calculates standard clustering evaluation metrics.

    This function computes several well-known metrics to evaluate the quality
    of the thematic clustering of document chunks.

    - Silhouette Score: Measures how similar an object is to its own cluster
      (cohesion) compared to other clusters (separation). Score ranges from
      -1 to 1, where a high value indicates dense and well-separated clusters.
    - Calinski-Harabasz Score: The ratio of between-cluster dispersion to
      within-cluster dispersion. A higher score indicates better-defined
      clusters.
    - Davies-Bouldin Score: The average similarity of each cluster with its
      most similar cluster. A lower score indicates better separation.
    - Local Density Index (LDI): Custom metric to assess local neighborhood
      purity.

    Args:
        embeddings (np.ndarray): The embeddings of the text chunks.
        labels (np.ndarray): The theme label for each chunk.

    Returns:
        Dict[str, float]: A dictionary with Silhouette, Calinski-Harabasz,
        Davies-Bouldin, and LDI scores.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(embeddings) <= len(unique_labels):
        return {
            "silhouette": -1.0,
            "calinski_harabasz": 0.0,
            "davies_bouldin": float("inf"),
            "local_density_index": 0.0,
        }

    return {
        "silhouette": float(silhouette_score(embeddings, labels, metric="cosine")),
        "calinski_harabasz": float(calinski_harabasz_score(embeddings, labels)),
        "davies_bouldin": float(davies_bouldin_score(embeddings, labels)),
        "local_density_index": float(local_density_index(embeddings, labels)),
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

    # Cohesion and Separation based metrics
    cohesion_sep_metrics = calculate_cohesion_separation(doc_embeddings, doc_labels)
    all_metrics.update(cohesion_sep_metrics)

    # Standard clustering metrics
    clustering_metrics = calculate_clustering_metrics(doc_embeddings, doc_labels)
    all_metrics.update(clustering_metrics)

    # Custom metrics
    all_metrics["internal_coherence_score"] = coherence_score(
        ref_embeddings, doc_embeddings
    )
    all_metrics["angular_stability"] = angular_stability(ref_embeddings, doc_embeddings)
    all_metrics["robustness_score"] = robustness_score(ref_embeddings, doc_embeddings)

    return all_metrics
