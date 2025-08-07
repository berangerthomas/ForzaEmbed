"""
Décomposition du score silhouette en cohésion intra-cluster et séparation inter-cluster
"""

from typing import Any, Dict

import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_samples, silhouette_score


def decompose_silhouette_score(
    embeddings: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    """
    Décompose le score silhouette en ses composantes a(i) et b(i).

    Le score silhouette s(i) = (b(i) - a(i)) / max(a(i), b(i))
    où:
    - a(i) = distance moyenne intra-cluster (cohésion) - PLUS BAS = MEILLEUR
    - b(i) = distance moyenne vers le cluster le plus proche - PLUS HAUT = MEILLEUR

    Args:
        embeddings: Matrice des embeddings (n_samples, n_features)
        labels: Labels des clusters (n_samples,)

    Returns:
        Dict contenant les moyennes de a(i), b(i) et le score silhouette
    """

    n_samples = len(embeddings)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Validation stricte : besoin d'au moins 2 clusters ET suffisamment d'échantillons
    if n_clusters < 2 or n_samples <= n_clusters:
        return {
            "mean_intra_cluster_distance": 0.0,
            "mean_inter_cluster_distance": 0.0,
            "silhouette_score": -1.0,
            "intra_cluster_quality": 0.0,  # 1 - normalized a(i)
            "inter_cluster_separation": 0.0,  # normalized b(i)
        }

    # Utiliser toujours 'cosine' pour l'analyse silhouette
    clustering_metric = "cosine"

    # Calcul de la matrice des distances
    distance_matrix = pairwise_distances(embeddings, metric=clustering_metric)

    n_samples = len(embeddings)
    a_values = []  # cohésion intra-cluster
    b_values = []  # séparation inter-cluster

    for i in range(n_samples):
        current_label = labels[i]

        # a(i): Distance moyenne intra-cluster
        same_cluster_mask = (labels == current_label) & (np.arange(n_samples) != i)
        if np.sum(same_cluster_mask) > 0:
            a_i = np.mean(distance_matrix[i][same_cluster_mask])
        else:
            a_i = 0.0
        a_values.append(a_i)

        # b(i): Distance moyenne vers le cluster le plus proche
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

    # Calcul du score silhouette pour vérification
    silhouette_computed = silhouette_score(embeddings, labels, metric=clustering_metric)

    # Normalisation pour obtenir des métriques interprétables (0-1)
    max_possible_distance = np.max(distance_matrix)

    # Qualité intra-cluster: 1 - (distance_moyenne / distance_max)
    # Plus proche de 1 = meilleure cohésion
    intra_quality = (
        1 - (np.mean(a_values) / max_possible_distance)
        if max_possible_distance > 0
        else 0
    )

    # Séparation inter-cluster: distance_moyenne / distance_max
    # Plus proche de 1 = meilleure séparation
    inter_separation = (
        np.mean(b_values) / max_possible_distance if max_possible_distance > 0 else 0
    )

    return {
        "mean_intra_cluster_distance": float(np.mean(a_values)),
        "mean_inter_cluster_distance": float(np.mean(b_values)),
        "silhouette_score": float(silhouette_computed),
        "intra_cluster_quality": float(intra_quality),  # 0-1, plus haut = meilleur
        "inter_cluster_separation": float(
            inter_separation
        ),  # 0-1, plus haut = meilleur
    }


def analyze_silhouette_by_cluster(
    embeddings: np.ndarray, labels: np.ndarray
) -> Dict[int, Dict[str, float]]:
    """
    Analyse détaillée du score silhouette par cluster.

    Args:
        embeddings: Matrice des embeddings
        labels: Labels des clusters

    Returns:
        Dict avec pour chaque cluster ses statistiques silhouette
    """

    n_samples = len(embeddings)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Validation stricte : même condition que decompose_silhouette_score
    if n_clusters < 2 or n_samples <= n_clusters:
        return {}

    sample_scores: np.ndarray = np.array(
        silhouette_samples(embeddings, labels, metric="cosine")
    )

    unique_labels = np.unique(labels)

    cluster_analysis = {}

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
) -> Dict[str, Any]:
    """
    Analyse complète du clustering avec décomposition silhouette.

    Note: Utilise toujours 'cosine' comme métrique pour l'analyse de clustering,
    indépendamment de la métrique de similarité utilisée pour l'évaluation des embeddings.

    Args:
        embeddings: Matrice des embeddings
        labels: Labels des clusters
        metric: Métrique de similarité (ignorée pour l'analyse silhouette)

    Returns:
        Analyse détaillée avec métriques globales et par cluster
    """

    global_decomp = decompose_silhouette_score(embeddings, labels)
    cluster_analysis = analyze_silhouette_by_cluster(embeddings, labels)

    return {"global_metrics": global_decomp, "cluster_analysis": cluster_analysis}
