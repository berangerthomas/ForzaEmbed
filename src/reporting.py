import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import extract_context_around_phrase


# Génère un fichier HTML représentant une heatmap de similarité.
def generate_heatmap_html(
    identifiant: str,
    nom: str,
    type_lieu: str,
    themes: list[str],
    phrases: list[str],
    similarites_norm: np.ndarray,
    cmap: LinearSegmentedColormap,
    output_dir: str,
    suffix: str,
) -> str:
    """
    Génère un fichier HTML de heatmap basé sur les scores de similarité.

    Args:
        identifiant (str): Identifiant du lieu.
        nom (str): Nom du lieu.
        type_lieu (str): Type de lieu.
        themes (list[str]): Thèmes utilisés.
        phrases (list[str]): Liste des phrases.
        similarites_norm (np.ndarray): Scores de similarité normalisés.
        cmap (LinearSegmentedColormap): Colormap pour la heatmap.
        output_dir (str): Dossier de sortie.
        suffix (str): Suffixe pour le nom de fichier.

    Returns:
        str: Chemin du fichier HTML généré.
    """
    couleurs = [cmap(score) for score in similarites_norm]
    html_output = f"<h2>{nom} ({type_lieu}) - Suffix: {suffix}</h2>\n"
    html_output += f"<p><strong>Thèmes utilisés:</strong> {', '.join(themes)}</p>\n"

    for phrase, score, couleur in zip(phrases, similarites_norm, couleurs):
        r, g, b, _ = [int(255 * x) for x in couleur]
        phrase_html = phrase.replace("\n", "<br>")
        tooltip_text = f"Similarité: {score:.3f}"
        html_output += f'<span style="background-color: rgb({r},{g},{b}); margin: 5px;" title="{tooltip_text}">{phrase_html}.</span> '

    filename = os.path.join(output_dir, f"{identifiant}{suffix}_heatmap.html")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_output)
    return filename


# Génère un fichier markdown filtré selon un seuil de similarité.
def generate_filtered_markdown(
    identifiant: str,
    nom: str,
    type_lieu: str,
    phrases: list[str],
    similarites_norm: np.ndarray,
    threshold: float,
    context_size: int,
    output_dir: str,
    suffix: str,
    model_name: str,
) -> str:
    """
    Génère un fichier markdown filtré selon les scores de similarité.

    Args:
        identifiant (str): Identifiant du lieu.
        nom (str): Nom du lieu.
        type_lieu (str): Type de lieu.
        phrases (list[str]): Liste des phrases.
        similarites_norm (np.ndarray): Scores de similarité normalisés.
        threshold (float): Seuil de filtrage.
        context_size (int): Taille de la fenêtre de contexte.
        output_dir (str): Dossier de sortie.
        suffix (str): Suffixe pour le nom de fichier.
        model_name (str): Nom du modèle utilisé.

    Returns:
        str: Chemin du fichier markdown généré.
    """
    relevant_indices = set()
    for i, score in enumerate(similarites_norm):
        if score >= threshold:
            start_idx = max(0, i - context_size)
            end_idx = min(len(phrases), i + context_size + 1)
            relevant_indices.update(range(start_idx, end_idx))

    relevant_phrases = [phrases[i] for i in sorted(list(relevant_indices))]

    content = f"# Résultat Filtré - {nom} ({type_lieu})\n\n"
    content += f"**Modèle:** {model_name}\n"
    content += f"**Seuil:** {threshold}\n"
    content += f"**Contexte:** {context_size}\n"
    content += f"**Phrases sélectionnées:** {len(relevant_phrases)}/{len(phrases)}\n\n"
    content += "## Contenu Filtré Brut\n\n"
    if relevant_phrases:
        content += "\n\n".join(relevant_phrases)
    else:
        content += "Aucune phrase pertinente trouvée."

    filename = os.path.join(output_dir, f"{identifiant}{suffix}_filtered.md")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return filename


# Génère un graphique radar pour comparer les modèles sur plusieurs métriques.
def generate_radar_chart(
    evaluation_results: dict[str, dict[str, float]], output_dir: str
) -> None:
    """
    Génère un graphique radar pour une comparaison globale des modèles.

    Args:
        evaluation_results (dict): Dictionnaire des résultats d'évaluation.
        output_dir (str): Dossier de sortie pour la visualisation.
    """
    model_names = list(evaluation_results.keys())
    if len(model_names) < 1:
        print("Not enough models to generate a radar chart.")
        return

    # Utiliser les mêmes métriques que les graphiques à barres
    metrics = [
        "cohesion",
        "separation",
        "discriminant_score",
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
    ]
    df = pd.DataFrame(evaluation_results).T[metrics]

    # Métriques où une valeur plus faible est meilleure
    lower_is_better = ["separation", "davies_bouldin"]

    # Normalisation des données
    normalized_df = df.copy()
    for metric in metrics:
        min_val = df[metric].min()
        max_val = df[metric].max()
        if max_val == min_val:
            normalized_df[metric] = 1.0  # Tous les modèles sont équivalents
        else:
            if metric in lower_is_better:
                # Inverser la normalisation : (max - x) / (max - min)
                normalized_df[metric] = (max_val - df[metric]) / (max_val - min_val)
            else:
                # Normalisation standard : (x - min) / (max - min)
                normalized_df[metric] = (df[metric] - min_val) / (max_val - min_val)

    # Création du graphique radar
    fig = go.Figure()
    for model_name in normalized_df.index:
        values = normalized_df.loc[model_name].tolist()
        # Fermer la boucle du radar
        values += values[:1]
        metric_labels = metrics + [metrics[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=metric_labels,
                fill="toself",
                name=model_name,
                hovertemplate=f"<b>{model_name}</b><br>"
                + "Métrique: %{theta}<br>"
                + "Score Normalisé: %{r:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
        title={
            "text": "<b>Comparaison Globale des Modèles (Score Normalisé)</b>",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20, "family": "Arial, sans-serif"},
        },
        legend=dict(
            title="Modèles",
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=100, b=100),
    )

    # Sauvegarder le graphique
    plot_filename = os.path.join(output_dir, "global_model_comparison_radar.png")
    try:
        fig.write_image(plot_filename, width=1200, height=800, scale=2)
        print(f"📊 Radar chart saved to: {plot_filename}")
    except Exception as e:
        print(f"❌ Could not save radar chart. Error: {e}")
        print(
            "Please ensure 'kaleido' is installed (`pip install kaleido`) for static image export."
        )


# Analyse et visualise les métriques de clustering pour différents modèles.
def analyze_and_visualize_clustering_metrics(
    evaluation_results: dict[str, dict[str, float]], output_dir: str
) -> None:
    """
    Analyse et visualise les métriques de clustering pour chaque modèle.

    Args:
        evaluation_results (dict): Dictionnaire des résultats d'évaluation.
        output_dir (str): Dossier de sortie pour la visualisation.
    """
    if not evaluation_results:
        print("No evaluation results to visualize.")
        return

    # Générer le graphique radar en premier
    generate_radar_chart(evaluation_results, output_dir)

    metrics = [
        "cohesion",
        "separation",
        "discriminant_score",
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
        "processing_time",
    ]
    titles = {
        "cohesion": "Cluster Cohesion (Higher is Better)",
        "separation": "Cluster Separation (Lower is Better)",
        "discriminant_score": "Discriminant Score (Higher is Better)",
        "silhouette": "Silhouette Score (Higher is Better)",
        "calinski_harabasz": "Calinski-Harabasz Index (Higher is Better)",
        "davies_bouldin": "Davies-Bouldin Index (Lower is Better)",
        "processing_time": "Processing Time (s) (Lower is Better)",
    }

    for metric in metrics:
        if metric not in evaluation_results[list(evaluation_results.keys())[0]]:
            continue
        model_names = list(evaluation_results.keys())
        values = [evaluation_results[model][metric] for model in model_names]

        # Sort models by metric value
        descending = metric not in [
            "separation",
            "davies_bouldin",
            "processing_time",
        ]
        sorted_indices = np.argsort(values)
        if descending:
            sorted_indices = sorted_indices[::-1]

        sorted_models = [model_names[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]

        try:
            plt.style.use("seaborn-v0_8-whitegrid")
            fig, ax = plt.subplots(figsize=(12, 7))
            palette = sns.color_palette("viridis", len(sorted_models))
            bars = ax.bar(sorted_models, sorted_values, color=palette)

            ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
            ax.set_title(titles[metric], fontsize=16, fontweight="bold")
            ax.tick_params(axis="x", rotation=45, labelsize=10)
            fig.autofmt_xdate()

            for bar in bars:
                yval = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    yval,
                    f"{yval:.3f}",
                    va="bottom" if yval >= 0 else "top",
                    ha="center",
                    fontsize=10,
                )

            plt.tight_layout(pad=2.0)
            plot_filename = os.path.join(output_dir, f"metric_{metric}_comparison.png")
            plt.savefig(plot_filename)
            print(f"📊 Metric comparison plot saved to: {plot_filename}")
            plt.close(fig)
        except Exception as e:
            print(f"❌ Could not generate plot for metric {metric}. Error: {e}")


# Analyse et visualise la variance des similarités d'embeddings pour différents modèles.
def analyze_and_visualize_variance(
    model_embeddings: dict[str, list[np.ndarray]], output_dir: str
) -> None:
    """
    Analyse et visualise la variance des similarités cosinus des embeddings pour chaque modèle.

    Args:
        model_embeddings (dict): Dictionnaire {nom_modèle: liste d'embeddings}.
        output_dir (str): Dossier de sortie pour la visualisation.
    """
    variances = {}
    print("\n--- Analyzing Embedding Variance ---")

    for model_name, embeddings_list in model_embeddings.items():
        # Filter out None or empty arrays before vstack
        valid_embeddings = [
            emb for emb in embeddings_list if emb is not None and emb.size > 0
        ]
        if not valid_embeddings:
            print(f"No valid embeddings found for model {model_name}, skipping.")
            continue

        # Concatenate all embeddings for the model into a single large array
        all_model_embeddings = np.vstack(valid_embeddings)
        print(
            f"Model {model_name}: Analyzing {all_model_embeddings.shape[0]} total embeddings."
        )

        # To avoid memory errors on very large datasets, sample if needed
        if all_model_embeddings.shape[0] > 2000:
            print("  - Sampling 2000 embeddings to manage memory.")
            indices = np.random.choice(
                all_model_embeddings.shape[0], 2000, replace=False
            )
            sample_embeddings = all_model_embeddings[indices]
        else:
            sample_embeddings = all_model_embeddings

        # Calculate pairwise cosine similarity
        similarity_matrix = cosine_similarity(sample_embeddings)

        # Use the upper triangle (excluding the diagonal) for variance calculation
        upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
        similarity_values = similarity_matrix[upper_triangle_indices]

        if similarity_values.size == 0:
            print("  - Not enough similarity values to calculate variance.")
            continue

        # Calculate the variance of these similarity scores
        variance = np.var(similarity_values)
        variances[model_name] = variance
        print(f"  - Variance of cosine similarities: {variance:.4f}")

    if not variances:
        print("No variances calculated. Cannot generate plot.")
        return

    # Identify the model with the highest variance (most contrast)
    best_model = max(variances, key=lambda k: variances[k])
    print(
        f"\n🏆 Model with highest contrast (max variance): {best_model} ({variances[best_model]:.4f})"
    )

    # --- Visualization ---
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 7))

        # Sort models by variance in descending order for the plot
        sorted_models = sorted(
            variances.keys(), key=lambda model: variances[model], reverse=True
        )
        values = [variances[model] for model in sorted_models]

        # Create a color palette
        palette = sns.color_palette("viridis", len(sorted_models))
        bars = ax.bar(sorted_models, values, color=palette)

        ax.set_ylabel("Variance of Cosine Similarities", fontsize=12)
        ax.set_title(
            "Embedding Contrast Analysis: Higher Variance is Better",
            fontsize=16,
            fontweight="bold",
        )
        ax.tick_params(axis="x", rotation=45, labelsize=10)
        # Ensure all labels are visible
        fig.autofmt_xdate()

        # Add value labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval,
                f"{yval:.4f}",
                va="bottom",
                ha="center",
                fontsize=10,
            )

        plt.tight_layout(pad=2.0)
        plot_filename = os.path.join(output_dir, "embedding_variance_comparison.png")
        plt.savefig(plot_filename)
        print(f"\n📊 Variance comparison plot saved to: {plot_filename}")
        plt.close(fig)  # Close the figure to free memory
    except Exception as e:
        print(f"❌ Could not generate plot. Error: {e}")


# Génère un rapport explicatif markdown basé sur les scores de similarité.
def generate_explanatory_markdown(
    identifiant: str,
    nom: str,
    type_lieu: str,
    phrases: list[str],
    similarites_norm: np.ndarray,
    themes: list[str],
    threshold: float,
    context_window: int,
    output_dir: str,
    suffix: str,
    model_name: str,
) -> str:
    """
    Génère un fichier markdown explicatif sur les zones à forte similarité.

    Args:
        identifiant (str): Identifiant du lieu.
        nom (str): Nom du lieu.
        type_lieu (str): Type de lieu.
        phrases (list[str]): Liste des phrases.
        similarites_norm (np.ndarray): Scores de similarité normalisés.
        themes (list[str]): Thèmes utilisés.
        threshold (float): Seuil de similarité.
        context_window (int): Fenêtre de contexte.
        output_dir (str): Dossier de sortie.
        suffix (str): Suffixe pour le nom de fichier.
        model_name (str): Nom du modèle utilisé.

    Returns:
        str: Chemin du fichier markdown généré.
    """
    hot_zones = []
    for i, (phrase, score) in enumerate(zip(phrases, similarites_norm)):
        if score >= threshold:
            hot_zones.append((i, phrase, score))

    hot_zones.sort(key=lambda x: x[2], reverse=True)

    markdown_content = f"# Rapport de Similarité - {nom} ({type_lieu})\n\n"
    markdown_content += f"**Identifiant:** {identifiant}\n"
    markdown_content += f"**Modèle:** {model_name}\n"
    markdown_content += f"**Thèmes recherchés:** {', '.join(themes)}\n"
    markdown_content += f"**Seuil de similarité:** {threshold}\n"
    markdown_content += f"**Fenêtre de contexte:** ±{context_window} phrases\n\n"

    if hot_zones:
        markdown_content += (
            f"## Zones à haute similarité ({len(hot_zones)} trouvées)\n\n"
        )
        for i, (phrase_idx, phrase, score) in enumerate(hot_zones, 1):
            markdown_content += f"### Zone {i} (Score: {score:.3f})\n\n"
            context = extract_context_around_phrase(phrases, phrase_idx, context_window)
            markdown_content += "**Contexte complet:**\n"
            markdown_content += f"> {context}\n\n"
            markdown_content += "**Phrase clé identifiée:**\n"
            markdown_content += f"> {phrase.strip()}\n\n---\n\n"
    else:
        markdown_content += "## Aucune zone à haute similarité trouvée\n\n"
        markdown_content += (
            f"Aucun segment n'a atteint le seuil de similarité de {threshold}.\n"
        )

    filename = os.path.join(output_dir, f"{identifiant}{suffix}_explanatory.md")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    return filename
