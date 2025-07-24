import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity


def calculate_cohesion_separation(
    embeddings: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """
    Calculates cohesion (intra-theme similarity) and separation (inter-theme similarity).

    Args:
        embeddings (np.ndarray): The embeddings of the text chunks.
        labels (np.ndarray): The theme label for each chunk.

    Returns:
        dict[str, float]: A dictionary containing cohesion, separation, and a combined discriminant score.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {
            "cohesion": 0.0,
            "separation": 0.0,
            "discriminant_score": 0.0,
        }

    # Cohesion: Average similarity within each theme
    cohesion_scores = []
    for label in unique_labels:
        theme_embeddings = embeddings[labels == label]
        if len(theme_embeddings) > 1:
            similarity_matrix = cosine_similarity(theme_embeddings)
            # Use upper triangle to avoid diagonal and duplicates
            indices = np.triu_indices_from(similarity_matrix, k=1)
            if similarity_matrix[indices].size > 0:
                cohesion_scores.append(np.mean(similarity_matrix[indices]))

    avg_cohesion = np.mean(cohesion_scores) if cohesion_scores else 0.0

    # Separation: Average similarity between different themes
    separation_scores = []
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            label1 = unique_labels[i]
            label2 = unique_labels[j]
            embeddings1 = embeddings[labels == label1]
            embeddings2 = embeddings[labels == label2]
            if embeddings1.size > 0 and embeddings2.size > 0:
                similarity_matrix = cosine_similarity(embeddings1, embeddings2)
                separation_scores.append(np.mean(similarity_matrix))

    avg_separation = np.mean(separation_scores) if separation_scores else 0.0

    # Discriminant Score
    discriminant_score = avg_cohesion / avg_separation if avg_separation > 0 else 0.0

    return {
        "cohesion": float(avg_cohesion),
        "separation": float(avg_separation),
        "discriminant_score": float(discriminant_score),
    }


def calculate_clustering_metrics(
    embeddings: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """
    Calculates standard clustering evaluation metrics.

    Args:
        embeddings (np.ndarray): The embeddings of the text chunks.
        labels (np.ndarray): The theme label for each chunk.

    Returns:
        dict[str, float]: A dictionary with Silhouette, Calinski-Harabasz, and Davies-Bouldin scores.
    """
    if len(np.unique(labels)) < 2 or len(embeddings) <= len(np.unique(labels)):
        return {
            "silhouette": -1.0,
            "calinski_harabasz": 0.0,
            "davies_bouldin": float("inf"),
        }

    return {
        "silhouette": float(silhouette_score(embeddings, labels, metric="cosine")),
        "calinski_harabasz": float(calinski_harabasz_score(embeddings, labels)),
        "davies_bouldin": float(davies_bouldin_score(embeddings, labels)),
    }
