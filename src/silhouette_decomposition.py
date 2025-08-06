"""
Décomposition du score silhouette en cohésion intra-cluster et séparation inter-cluster
"""

from typing import Dict

import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_samples, silhouette_score


def decompose_silhouette_score(
    embeddings: np.ndarray, labels: np.ndarray, metric: str = "cosine"
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
        metric: Métrique de distance ('cosine', 'euclidean', etc.)

    Returns:
        Dict contenant les moyennes de a(i), b(i) et le score silhouette
    """

    if len(np.unique(labels)) < 2:
        return {
            "mean_intra_cluster_distance": 0.0,
            "mean_inter_cluster_distance": 0.0,
            "silhouette_score": -1.0,
            "intra_cluster_quality": 0.0,  # 1 - normalized a(i)
            "inter_cluster_separation": 0.0,  # normalized b(i)
        }

    # Calcul de la matrice des distances
    distance_matrix = pairwise_distances(embeddings, metric=metric)

    n_samples = len(embeddings)
    a_values = []  # cohésion intra-cluster
    b_values = []  # séparation inter-cluster

    unique_labels = np.unique(labels)

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
    silhouette_computed = silhouette_score(embeddings, labels, metric=metric)

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
    embeddings: np.ndarray, labels: np.ndarray, metric: str = "cosine"
) -> Dict[int, Dict[str, float]]:
    """
    Analyse détaillée du score silhouette par cluster.

    Returns:
        Dict avec pour chaque cluster ses statistiques silhouette
    """

    if len(np.unique(labels)) < 2:
        return {}

    sample_scores = silhouette_samples(embeddings, labels, metric=metric)
    unique_labels = np.unique(labels)

    cluster_analysis = {}

    for label in unique_labels:
        cluster_mask = labels == label
        cluster_silhouettes = sample_scores[cluster_mask]

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
    embeddings: np.ndarray, labels: np.ndarray, metric: str = "cosine"
) -> Dict[str, any]:
    """
    Analyse complète du clustering avec décomposition silhouette.

    Returns:
        Analyse détaillée avec métriques globales et par cluster
    """

    global_decomp = decompose_silhouette_score(embeddings, labels, metric)
    cluster_analysis = analyze_silhouette_by_cluster(embeddings, labels, metric)

    # Diagnostics supplémentaires
    diagnostics = {}

    if len(cluster_analysis) > 0:
        # Quel facteur limite le plus le score ?
        intra_quality = global_decomp["intra_cluster_quality"]
        inter_separation = global_decomp["inter_cluster_separation"]

        if intra_quality < inter_separation:
            limiting_factor = "cohésion intra-cluster"
            improvement_suggestion = "Améliorer la compacité des clusters"
        else:
            limiting_factor = "séparation inter-cluster"
            improvement_suggestion = "Améliorer la séparation entre clusters"

        diagnostics = {
            "limiting_factor": limiting_factor,
            "improvement_suggestion": improvement_suggestion,
            "balance_ratio": intra_quality / inter_separation
            if inter_separation > 0
            else 0,
        }

        # Identification des clusters problématiques
        problematic_clusters = []
        for cluster_id, stats in cluster_analysis.items():
            if stats["mean_silhouette"] < 0.3:  # Seuil arbitraire
                problematic_clusters.append(cluster_id)

        diagnostics["problematic_clusters"] = problematic_clusters

    return {
        "global_metrics": global_decomp,
        "cluster_analysis": cluster_analysis,
        "diagnostics": diagnostics,
    }


# Exemple d'utilisation avec vérification
if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    # Données de test
    X, y = make_blobs(
        n_samples=3000, centers=2, n_features=2, random_state=42, cluster_std=1.5
    )

    # VÉRIFICATION: Comparaison avec sklearn
    sklearn_silhouette = silhouette_score(X, y, metric="euclidean")

    # Notre analyse décomposée
    analysis = enhanced_silhouette_analysis(X, y, metric="euclidean")
    our_silhouette = analysis["global_metrics"]["silhouette_score"]

    print("=== VÉRIFICATION DES CALCULS ===")
    print(f"Score silhouette sklearn:      {sklearn_silhouette:.6f}")
    print(f"Score silhouette notre calcul: {our_silhouette:.6f}")
    print(
        f"Différence absolue:            {abs(sklearn_silhouette - our_silhouette):.6f}"
    )
    print(
        f"Calculs identiques: {np.isclose(sklearn_silhouette, our_silhouette, atol=1e-10)}"
    )

    print("\n=== ANALYSE SILHOUETTE DÉCOMPOSÉE ===")
    print(
        f"Score silhouette global: {analysis['global_metrics']['silhouette_score']:.3f}"
    )
    print(
        f"Distance intra-cluster moyenne (a): {analysis['global_metrics']['mean_intra_cluster_distance']:.3f}"
    )
    print(
        f"Distance inter-cluster moyenne (b): {analysis['global_metrics']['mean_inter_cluster_distance']:.3f}"
    )
    print(
        f"Qualité cohésion intra-cluster: {analysis['global_metrics']['intra_cluster_quality']:.3f}"
    )
    print(
        f"Qualité séparation inter-cluster: {analysis['global_metrics']['inter_cluster_separation']:.3f}"
    )
    print(f"Facteur limitant: {analysis['diagnostics']['limiting_factor']}")
    print(f"Suggestion: {analysis['diagnostics']['improvement_suggestion']}")

    print("\n=== ANALYSE PAR CLUSTER ===")
    for cluster_id, stats in analysis["cluster_analysis"].items():
        print(
            f"Cluster {cluster_id}: silhouette={stats['mean_silhouette']:.3f}, "
            f"taille={stats['size']}, positifs={stats['proportion_positive']:.1%}"
        )

    # Test avec différentes métriques pour vérifier la robustesse
    print("\n=== TEST AUTRES MÉTRIQUES ===")
    for metric in ["euclidean", "cosine", "manhattan"]:
        try:
            sklearn_score = silhouette_score(X, y, metric=metric)
            our_analysis = enhanced_silhouette_analysis(X, y, metric=metric)
            our_score = our_analysis["global_metrics"]["silhouette_score"]
            diff = abs(sklearn_score - our_score)
            print(
                f"{metric:>10}: sklearn={sklearn_score:.4f}, notre={our_score:.4f}, diff={diff:.6f}"
            )
        except Exception as e:
            print(f"{metric:>10}: ERREUR - {e}")
