import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.metrics.pairwise import cosine_similarity

# --- Métrique 1: Score de Séparation des Classes ---


def calculate_separation_score(embeddings_pertinents, embeddings_non_pertinents):
    """
    Calcule le score de séparation basé sur la similarité intra-classe et inter-classe.
    """
    if len(embeddings_pertinents) < 2 or len(embeddings_non_pertinents) == 0:
        return 0, 0, 0

    # Similarité intra-classe (entre phrases pertinentes)
    sim_intra_matrix = cosine_similarity(embeddings_pertinents)
    # On utilise le triangle supérieur de la matrice sans la diagonale
    indices_triu = np.triu_indices_from(sim_intra_matrix, k=1)
    sim_intra = (
        np.mean(sim_intra_matrix[indices_triu]) if indices_triu[0].size > 0 else 0
    )

    # Similarité inter-classe (entre pertinentes et non pertinentes)
    sim_inter_matrix = cosine_similarity(
        embeddings_pertinents, embeddings_non_pertinents
    )
    sim_inter = np.mean(sim_inter_matrix)

    separation_score = sim_intra - sim_inter
    return separation_score, sim_intra, sim_inter


def plot_separation_scores(results, output_dir):
    """
    Génère un graphique en barres groupées pour les scores de séparation,
    et une courbe montrant la différence (sim_intra - sim_inter) pour chaque modèle.
    Plus cette différence est grande, meilleur est le modèle.
    Les modèles sont triés par ordre décroissant de la différence (intra - inter).
    """
    # Calculer les scores et différences
    model_names = list(results.keys())
    sim_intra_scores = [res["sim_intra"] for res in results.values()]
    sim_inter_scores = [res["sim_inter"] for res in results.values()]
    separation_diffs = [
        intra - inter for intra, inter in zip(sim_intra_scores, sim_inter_scores)
    ]

    # Trier par ordre décroissant de la différence
    sorted_items = sorted(
        zip(model_names, sim_intra_scores, sim_inter_scores, separation_diffs),
        key=lambda x: x[3],
        reverse=True,
    )
    model_names, sim_intra_scores, sim_inter_scores, separation_diffs = zip(
        *sorted_items
    )

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(14, 8))

    rects1 = ax1.bar(
        x - width / 2,
        sim_intra_scores,
        width,
        label="Similarité Intra-Classe (Pertinentes)",
        color="g",
    )
    rects2 = ax1.bar(
        x + width / 2,
        sim_inter_scores,
        width,
        label="Similarité Inter-Classe (Pert./Non-Pert.)",
        color="r",
    )

    ax1.set_ylabel("Similarité Cosinus Moyenne")
    ax1.set_title("Score de Séparation des Classes par Modèle")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha="right")
    ax1.legend(loc="upper left")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Ajout de la courbe de différence
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        separation_diffs,
        color="b",
        marker="o",
        linestyle="-",
        linewidth=2,
        label="Différence (Intra - Inter)",
    )
    ax2.set_ylabel("Différence de Similarité", color="b")
    ax2.tick_params(axis="y", labelcolor="b")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "separation_scores_comparison.png"))
    plt.close(fig)


# --- Métrique 2: Métriques de Classification ---


def calculate_classification_metrics(y_true, y_scores):
    """
    Calcule l'AUC de la courbe ROC et le score AP de la courbe Précision-Rappel.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "average_precision": avg_precision,
    }


def plot_roc_curves(results, output_dir):
    """
    Génère un graphique avec les courbes ROC de tous les modèles.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    for model_name, metrics in results.items():
        if "roc_auc" in metrics:
            ax.plot(
                metrics["fpr"],
                metrics["tpr"],
                label=f"{model_name} (AUC = {metrics['roc_auc']:.3f})",
            )

    ax.plot([0, 1], [0, 1], "k--", label="Aléatoire")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Taux de Faux Positifs")
    ax.set_ylabel("Taux de Vrais Positifs")
    ax.set_title("Courbes ROC par Modèle")
    ax.legend(loc="lower right")
    ax.grid(True)

    plt.savefig(os.path.join(output_dir, "roc_curves_comparison.png"))
    plt.close(fig)


# --- Métrique 3: Distance des Centroïdes et Visualisation t-SNE ---


def calculate_centroid_distance(embeddings_pertinents, embeddings_non_pertinents):
    """
    Calcule la distance cosinus entre les centroïdes des deux classes.
    """
    if len(embeddings_pertinents) == 0 or len(embeddings_non_pertinents) == 0:
        return 0

    centroid_pertinent = np.mean(embeddings_pertinents, axis=0).reshape(1, -1)
    centroid_non_pertinent = np.mean(embeddings_non_pertinents, axis=0).reshape(1, -1)

    # La similarité cosinus est de 1 - distance cosinus
    distance = 1 - cosine_similarity(centroid_pertinent, centroid_non_pertinent)[0][0]
    return distance


def plot_centroid_distances(results, output_dir):
    """
    Génère un graphique en barres pour les distances des centroïdes,
    trié par ordre décroissant.
    """
    # Trier les modèles par distance décroissante
    sorted_items = sorted(
        results.items(), key=lambda item: item[1]["centroid_distance"], reverse=True
    )
    model_names = [item[0] for item in sorted_items]
    distances = [item[1]["centroid_distance"] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x=model_names, y=distances, palette="viridis", ax=ax)

    ax.set_ylabel("Distance Cosinus")
    ax.set_title("Distance entre les Centroïdes des Classes par Modèle")
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "centroid_distances_comparison.png"))
    plt.close(fig)


def plot_tsne_projection(embeddings, labels, model_name, output_dir):
    """
    Génère une projection t-SNE 2D des embeddings.
    """
    if len(embeddings) < 2:
        print(f"Skipping t-SNE for {model_name}: not enough data points.")
        return

    tsne = TSNE(
        n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1)
    )
    embeddings_2d = tsne.fit_transform(embeddings)

    df = pd.DataFrame(embeddings_2d, columns=["x", "y"])
    df["label"] = ["Pertinent" if l else "Non Pertinent" for l in labels]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="label",
        style="label",
        palette={"Pertinent": "g", "Non Pertinent": "r"},
        ax=ax,
    )

    ax.set_title(f"Projection t-SNE des Embeddings - Modèle: {model_name}")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(title="Classe")
    ax.grid(True, linestyle="--", alpha=0.6)

    # Crée un sous-dossier pour les projections t-SNE pour ne pas polluer le dossier principal
    tsne_output_dir = os.path.join(output_dir, "tsne_projections")
    os.makedirs(tsne_output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(tsne_output_dir, f"tsne_{model_name.replace('/', '_')}.png")
    )
    plt.close(fig)
