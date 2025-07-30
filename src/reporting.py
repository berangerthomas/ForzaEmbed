import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import extract_context_around_phrase


# Generates an HTML file representing a similarity heatmap.
def generate_heatmap_html(
    identifiant: str,
    nom: str,
    type_lieu: str,
    themes: list[str],
    phrases: list[str],
    similarites_norm: np.ndarray,
    cmap: LinearSegmentedColormap,
    output_dir: str,
    model_name: str,
    run_name: str,
) -> str:
    """
    Generates an HTML heatmap file based on similarity scores.

    Args:
        identifiant (str): Location identifier.
        nom (str): Location name.
        type_lieu (str): Location type.
        themes (list[str]): Used themes.
        phrases (list[str]): List of sentences.
        similarites_norm (np.ndarray): Normalized similarity scores.
        cmap (LinearSegmentedColormap): Colormap for the heatmap.
        output_dir (str): Output directory.
        model_name (str): Base model name.
        run_name (str): Full run name for the filename.

    Returns:
        str: Path to the generated HTML file.
    """
    couleurs = [cmap(score) for score in similarites_norm]
    html_output = f"<h2>{nom} ({type_lieu})</h2>\n"
    html_output += f"<p><strong>Run:</strong> {run_name}<br>"
    html_output += f"<strong>Base model:</strong> {model_name}<br>"
    html_output += f"<strong>Used themes:</strong> {', '.join(themes)}</p>\n"

    for phrase, score, couleur in zip(phrases, similarites_norm, couleurs):
        r, g, b, _ = [int(255 * x) for x in couleur]
        phrase_html = phrase.replace("\n", "<br>")
        tooltip_text = f"Similarity: {score:.3f}"
        html_output += f'<span style="background-color: rgb({r},{g},{b}); margin: 5px;" title="{tooltip_text}">{phrase_html}.</span> '

    safe_run_name = run_name.replace("/", "_")
    filename = os.path.join(output_dir, f"{identifiant}_{safe_run_name}_heatmap.html")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_output)
    return filename


# Generates a markdown file filtered by a similarity threshold.
def generate_filtered_markdown(
    identifiant: str,
    nom: str,
    type_lieu: str,
    phrases: list[str],
    similarites_norm: np.ndarray,
    threshold: float,
    output_dir: str,
    model_name: str,
    run_name: str,
) -> str:
    """
    Generates a markdown file filtered by similarity scores.

    Args:
        identifiant (str): Location identifier.
        nom (str): Location name.
        type_lieu (str): Location type.
        phrases (list[str]): List of sentences.
        similarites_norm (np.ndarray): Normalized similarity scores.
        threshold (float): Filtering threshold.
        output_dir (str): Output directory.
        model_name (str): Base model name.
        run_name (str): Full run name for the filename.

    Returns:
        str: Path to the generated markdown file.
    """
    relevant_indices = set()
    for i, score in enumerate(similarites_norm):
        if score >= threshold:
            relevant_indices.add(i)

    relevant_phrases = [phrases[i] for i in sorted(list(relevant_indices))]

    content = f"# Filtered Result - {nom} ({type_lieu})\n\n"
    content += f"**Run:** {run_name}\n"
    content += f"**Base model:** {model_name}\n"
    content += f"**Threshold:** {threshold}\n"
    content += f"**Selected sentences:** {len(relevant_phrases)}/{len(phrases)}\n\n"
    content += "## Raw Filtered Content\n\n"
    if relevant_phrases:
        content += "\n\n".join(relevant_phrases)
    else:
        content += "No relevant sentence found."

    safe_run_name = run_name.replace("/", "_")
    filename = os.path.join(output_dir, f"{identifiant}_{safe_run_name}_filtered.md")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return filename


# Generates a radar chart to compare models on several metrics.
def generate_radar_chart(
    evaluation_results: dict[str, dict[str, float]], output_dir: str
) -> str | None:
    """
    Generates a radar chart for a global comparison of models.

    Args:
        evaluation_results (dict): Evaluation results dictionary.
        output_dir (str): Output directory for the visualization.
    """
    model_names = list(evaluation_results.keys())
    if len(model_names) < 1:
        print("Not enough models to generate a radar chart.")
        return

    # Use the same metrics as the bar charts
    metrics = [
        "cohesion",
        "separation",
        "discriminant_score",
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
    ]
    df = pd.DataFrame(evaluation_results).T[metrics]

    # Metrics where a lower value is better
    lower_is_better = ["separation", "davies_bouldin"]

    # Data normalization
    normalized_df = df.copy()
    for metric in metrics:
        min_val = df[metric].min()
        max_val = df[metric].max()
        if max_val == min_val:
            normalized_df[metric] = 1.0  # All models are equivalent
        else:
            if metric in lower_is_better:
                # Inverse normalization: (max - x) / (max - min)
                normalized_df[metric] = (max_val - df[metric]) / (max_val - min_val)
            else:
                # Standard normalization: (x - min) / (max - min)
                normalized_df[metric] = (df[metric] - min_val) / (max_val - min_val)

    # Radar chart creation
    fig = go.Figure()
    for model_name in normalized_df.index:
        values = normalized_df.loc[model_name].tolist()
        # Close the radar loop
        values += values[:1]
        metric_labels = metrics + [metrics[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=metric_labels,
                fill="toself",
                name=model_name,
                hovertemplate=f"<b>{model_name}</b><br>"
                + "Metric: %{theta}<br>"
                + "Normalized Score: %{r:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
        title={
            "text": "<b>Global Model Comparison (Normalized Score)</b>",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20, "family": "Arial, sans-serif"},
        },
        legend=dict(
            title="Models",
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=100, b=100),
    )

    # Save the chart
    plot_filename = os.path.join(output_dir, "global_model_comparison_radar.png")
    try:
        fig.write_image(plot_filename, width=9600, height=6400, scale=2)
        print(f"📊 Radar chart saved to: {plot_filename}")
        return plot_filename
    except Exception as e:
        print(f"❌ Could not save radar chart. Error: {e}")
        print(
            "Please ensure 'kaleido' is installed (`pip install kaleido`) for static image export."
        )
        return None


# Analyzes and visualizes clustering metrics for different models.
def analyze_and_visualize_clustering_metrics(
    evaluation_results: dict[str, dict[str, float]], output_dir: str
) -> str | None:
    """
    Analyzes and visualizes clustering metrics for each model.

    Args:
        evaluation_results (dict): Evaluation results dictionary.
        output_dir (str): Output directory for the visualization.
    """
    if not evaluation_results:
        print("No evaluation results to visualize.")
        return None

    # Generate the radar chart first
    radar_chart_path = generate_radar_chart(evaluation_results, output_dir)

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
            fig, ax = plt.subplots(figsize=(96, 56))
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
    return radar_chart_path


# Analyzes and visualizes the variance of embedding similarities for different models.
def analyze_and_visualize_variance(
    model_embeddings: dict[str, list[np.ndarray]], output_dir: str
) -> str | None:
    """
    Analyzes and visualizes the variance of cosine similarities of embeddings for each model.

    Args:
        model_embeddings (dict): Dictionary {model_name: list of embeddings}.
        output_dir (str): Output directory for the visualization.
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
        return None

    # Identify the model with the highest variance (most contrast)
    best_model = max(variances, key=lambda k: variances[k])
    print(
        f"\n🏆 Model with highest contrast (max variance): {best_model} ({variances[best_model]:.4f})"
    )

    # --- Visualization ---
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(96, 56))

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
        return plot_filename
    except Exception as e:
        print(f"❌ Could not generate plot. Error: {e}")
        return None


# Generates an explanatory markdown report based on similarity scores.
def generate_explanatory_markdown(
    identifiant: str,
    nom: str,
    type_lieu: str,
    phrases: list[str],
    similarites_norm: np.ndarray,
    themes: list[str],
    threshold: float,
    output_dir: str,
    model_name: str,
    run_name: str,
) -> str:
    """
    Generates an explanatory markdown file about high similarity zones.

    Args:
        identifiant (str): Location identifier.
        nom (str): Location name.
        type_lieu (str): Location type.
        phrases (list[str]): List of sentences.
        similarites_norm (np.ndarray): Normalized similarity scores.
        themes (list[str]): Used themes.
        threshold (float): Similarity threshold.
        output_dir (str): Output directory.
        model_name (str): Base model name.
        run_name (str): Full run name for the filename.

    Returns:
        str: Path to the generated markdown file.
    """
    hot_zones = []
    for i, (phrase, score) in enumerate(zip(phrases, similarites_norm)):
        if score >= threshold:
            hot_zones.append((i, phrase, score))

    hot_zones.sort(key=lambda x: x[2], reverse=True)

    markdown_content = f"# Similarity Report - {nom} ({type_lieu})\n\n"
    markdown_content += f"**Identifier:** {identifiant}\n"
    markdown_content += f"**Run:** {run_name}\n"
    markdown_content += f"**Base model:** {model_name}\n"
    markdown_content += f"**Searched themes:** {', '.join(themes)}\n"
    markdown_content += f"**Similarity threshold:** {threshold}\n\n"

    if hot_zones:
        markdown_content += f"## High similarity zones ({len(hot_zones)} found)\n\n"
        for i, (phrase_idx, phrase, score) in enumerate(hot_zones, 1):
            markdown_content += f"### Zone {i} (Score: {score:.3f})\n\n"
            context = extract_context_around_phrase(phrases, phrase_idx)
            markdown_content += "**Full context:**\n"
            markdown_content += f"> {context}\n\n"
            markdown_content += "**Identified key sentence:**\n"
            markdown_content += f"> {phrase.strip()}\n\n---\n\n"
    else:
        markdown_content += "## No high similarity zone found\n\n"
        markdown_content += (
            f"No segment reached the similarity threshold of {threshold}.\n"
        )

    safe_run_name = run_name.replace("/", "_")
    filename = os.path.join(output_dir, f"{identifiant}_{safe_run_name}_explanatory.md")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    return filename


# Generates a t-SNE visualization of embeddings.
def generate_tsne_visualization(
    model_embeddings: dict[str, list[np.ndarray]],
    labels: list[np.ndarray],
    themes: list[str],
    output_dir: str,
    perplexity: int = 30,
    consolidate_themes: bool = True,
) -> dict[str, str]:
    """
    Generates a t-SNE visualization for each model.

    Args:
        model_embeddings (dict): Dictionary {model_name: list of embeddings}.
        labels (list[np.ndarray]): List of labels for each embedding.
        themes (list[str]): List of original themes.
        output_dir (str): Output directory.
        perplexity (int): Perplexity for the t-SNE algorithm.
        consolidate_themes (bool): If True, groups all themes as 'horaires'.
    """
    print("\n--- Generating t-SNE Visualizations ---")

    plot_paths = {}

    # Flatten the list of label arrays into a single list of integers
    if isinstance(labels[0], np.ndarray):
        flat_labels = np.concatenate(labels).ravel()
    else:
        flat_labels = np.array(labels)

    if consolidate_themes:
        display_labels = []
        for label_index in flat_labels:
            if themes[label_index] == "autre":
                display_labels.append("autre")
            else:
                display_labels.append("horaires")
    else:
        # Use original themes and labels
        display_labels = [themes[i] for i in flat_labels]

    unique_display_labels = sorted(list(set(display_labels)))
    palette = sns.color_palette("husl", len(unique_display_labels))
    label_to_color = {
        label: color for label, color in zip(unique_display_labels, palette)
    }

    for model_name, embeddings_list in model_embeddings.items():
        valid_embeddings = [
            emb for emb in embeddings_list if emb is not None and emb.size > 0
        ]
        if not valid_embeddings:
            print(f"Skipping t-SNE for {model_name}: no valid embeddings.")
            continue

        all_embeddings = np.vstack(valid_embeddings)
        if all_embeddings.shape[0] != len(flat_labels):
            print(
                f"Skipping t-SNE for {model_name}: mismatch between embeddings count ({all_embeddings.shape[0]}) and labels count ({len(flat_labels)})."
            )
            continue

        print(f"Processing t-SNE for {model_name}...")
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(perplexity, all_embeddings.shape[0] - 1),
            learning_rate="auto",
            init="pca",
            n_jobs=-1,
        )
        tsne_results = tsne.fit_transform(all_embeddings)

        df = pd.DataFrame(
            {
                "tsne-2d-one": tsne_results[:, 0],
                "tsne-2d-two": tsne_results[:, 1],
                "label": display_labels,
            }
        )

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(128, 96))
        sns.scatterplot(
            x="tsne-2d-one",
            y="tsne-2d-two",
            hue="label",
            palette=label_to_color,
            data=df,
            ax=ax,
            s=50,
            alpha=0.7,
        )

        ax.set_title(
            f"t-SNE Visualization of Embeddings for {model_name}",
            fontsize=18,
            fontweight="bold",
        )
        ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
        ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
        ax.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout(rect=(0, 0, 0.85, 1))
        plot_filename = os.path.join(output_dir, f"tsne_{model_name}.png")
        try:
            plt.savefig(plot_filename)
            print(f"📊 t-SNE plot for {model_name} saved to: {plot_filename}")
            plot_paths[model_name] = plot_filename
        except Exception as e:
            print(f"❌ Could not save t-SNE plot for {model_name}. Error: {e}")
        finally:
            plt.close(fig)
    return plot_paths
