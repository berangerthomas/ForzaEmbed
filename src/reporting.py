import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import extract_context_around_phrase


# Generates data for a t-SNE scatter plot to visualize theme separation.
def generate_similarity_scatterplot_data(
    identifiant: str,
    run_name: str,
    embeddings: np.ndarray,
    similarities: np.ndarray,
    threshold: float,
) -> dict | None:
    """
    Generates data for a t-SNE scatter plot of embeddings, colored by similarity.

    Args:
        identifiant (str): Unique identifier for the document.
        run_name (str): The name of the processing run.
        embeddings (np.ndarray): The embedding vectors for the document's phrases.
        similarities (np.ndarray): The similarity scores for each phrase.
        threshold (float): The similarity threshold for coloring points.

    Returns:
        dict | None: A dictionary containing the data for the plot, or None.
    """
    if embeddings is None or embeddings.shape[0] < 2:
        return None

    try:
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(30, embeddings.shape[0] - 1),
            learning_rate="auto",
            init="pca",
            n_jobs=-1,
        )
        tsne_results = tsne.fit_transform(embeddings)
    except Exception as e:
        print(f"Error during t-SNE for {identifiant} in run {run_name}: {e}")
        return None

    labels = [
        f"Above Threshold ({threshold})" if score >= threshold else "Below Threshold"
        for score in similarities
    ]

    return {
        "x": tsne_results[:, 0].tolist(),
        "y": tsne_results[:, 1].tolist(),
        "labels": labels,
        "similarities": similarities.tolist(),
        "title": f"Theme Separation for {identifiant}<br>Run: {run_name}",
        "threshold": threshold,
    }


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

    if relevant_phrases:
        content = "\n\n".join(relevant_phrases)
    else:
        content = "No relevant sentence found."

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

    metrics = [
        "cohesion",
        "separation",
        "discriminant_score",
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
    ]

    # Convert dict to a list of records for Polars, which is more robust
    records = [
        {"model_name": model, **results}
        for model, results in evaluation_results.items()
    ]
    df = pl.DataFrame(records)

    # Ensure all metrics are present, fill with a default (e.g., 0 or NaN) if not
    for metric in metrics:
        if metric not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(metric))

    # Select only the metrics we need for the chart, plus model_name
    df = df.select(["model_name"] + metrics)

    lower_is_better = ["separation", "davies_bouldin"]

    # Data normalization
    normalized_df = df.clone().fill_null(0)  # Fill nulls before normalization

    normalization_exprs = []
    for metric in metrics:
        min_val = normalized_df[metric].min()
        max_val = normalized_df[metric].max()

        col_expr = pl.col(metric)

        if min_val is None or max_val is None or max_val == min_val:
            expr = pl.lit(0.5).alias(metric)
        else:
            # For metrics where lower is better, the score is inverted.
            # (max - value) / (max - min)
            if metric in lower_is_better:
                expr = (
                    (pl.lit(max_val) - col_expr) / (pl.lit(max_val) - pl.lit(min_val))
                ).alias(metric)
            # For metrics where higher is better, the score is standard.
            # (value - min) / (max - min)
            else:
                expr = (
                    (col_expr - pl.lit(min_val)) / (pl.lit(max_val) - pl.lit(min_val))
                ).alias(metric)
        normalization_exprs.append(expr)

    if normalization_exprs:
        normalized_df = normalized_df.with_columns(normalization_exprs)

    # Radar chart creation
    fig = go.Figure()

    # Select only numeric columns for plotting
    numeric_df = normalized_df.select(metrics)

    for i, model_name in enumerate(model_names):
        values = numeric_df.row(i)
        values_list = list(values) + [values[0]]  # Loop back
        metric_labels = metrics + [metrics[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values_list,
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
        fig.write_image(plot_filename, width=1200, height=800, scale=2)
        print(f"📊 Radar chart saved to: {plot_filename}")
        return plot_filename
    except Exception as e:
        print(f"❌ Could not save radar chart. Error: {e}")
        return None


# Analyzes and visualizes clustering metrics for different models.
def analyze_and_visualize_clustering_metrics(
    evaluation_results: dict[str, dict[str, float]],
    output_dir: str,
    top_n: int | None = None,
) -> str | None:
    """
    Analyzes and visualizes clustering metrics for each model.

    Args:
        evaluation_results (dict): Evaluation results dictionary.
        output_dir (str): Output directory for the visualization.
        top_n (int, optional): If set, displays only the top N models for each metric.
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

    # Convert the results to a Polars DataFrame for robust manipulation
    records = [
        {"model_name": model, **results}
        for model, results in evaluation_results.items()
    ]
    df = pl.DataFrame(records)

    for metric in metrics:
        if metric not in df.columns:
            continue

        # Filter out models where the metric is null
        metric_df = df.filter(pl.col(metric).is_not_null())

        if metric_df.height == 0:
            continue

        # Sort the DataFrame by the metric value
        descending = metric not in [
            "separation",
            "davies_bouldin",
            "processing_time",
        ]
        sorted_df = metric_df.sort(metric, descending=descending)

        # If top_n is specified, slice the DataFrame
        if top_n:
            sorted_df = sorted_df.head(top_n)

        if sorted_df.height == 0:
            continue

        try:
            plt.style.use("seaborn-v0_8-whitegrid")
            fig, ax = plt.subplots(figsize=(70, 40))  # Increased figure size

            # --- Data Preparation for Plotting ---
            plot_data = sorted_df.to_pandas()

            # Wrap model names for better display
            # Adjust the wrap width as needed
            plot_data["wrapped_model_name"] = plot_data["model_name"].str.wrap(30)

            # --- Plotting ---
            barplot = sns.barplot(
                x="wrapped_model_name",
                y=metric,
                data=plot_data,
                hue="model_name",
                palette="viridis",
                ax=ax,
                legend=False,
                dodge=False,  # Ensure bars are not dodged
            )

            # --- Aesthetics and Labels ---
            ax.set_xlabel("Model", fontsize=12, fontweight="bold")
            ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
            ax.set_title(titles[metric], fontsize=16, fontweight="bold")

            # Set x-tick labels explicitly to ensure they match the bars
            ax.set_xticks(range(len(plot_data)))
            ax.set_xticklabels(
                plot_data["wrapped_model_name"],
                rotation=45,
                ha="right",
                fontsize=10,
            )

            # Add value labels on top of bars
            for i, v in enumerate(sorted_df[metric]):
                ax.text(
                    i,
                    v,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

            # --- Layout and Saving ---
            # Adjust bottom margin to make space for labels
            plt.subplots_adjust(bottom=0.25)
            plt.tight_layout(pad=3.0)

            plot_filename = os.path.join(output_dir, f"metric_{metric}_comparison.png")
            plt.savefig(plot_filename, bbox_inches="tight")
            print(f"📊 Metric comparison plot saved to: {plot_filename}")
            plt.close(fig)
        except Exception as e:
            print(f"❌ Could not generate plot for metric {metric}. Error: {e}")

    return radar_chart_path


# Analyzes and visualizes the variance of embedding similarities for different models.
def analyze_and_visualize_variance(
    model_embeddings: dict[str, list[np.ndarray]],
    output_dir: str,
    top_n: int | None = None,
) -> str | None:
    """
    Analyzes and visualizes the variance of cosine similarities of embeddings for each model.

    Args:
        model_embeddings (dict): Dictionary {model_name: list of embeddings}.
        output_dir (str): Output directory for the visualization.
        top_n (int, optional): If set, displays only the top N models.
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
        # print(
        #     f"Model {model_name}: Analyzing {all_model_embeddings.shape[0]} total embeddings."
        # )

        # To avoid memory errors on very large datasets, sample if needed
        if all_model_embeddings.shape[0] > 20000000:
            print("  - Sampling 20000000 embeddings to manage memory.")
            indices = np.random.choice(
                all_model_embeddings.shape[0], 20000000, replace=False
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
        # print(f"  - Variance of cosine similarities: {variance:.4f}")

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
        # Convert variances to a Polars DataFrame
        df = pl.DataFrame(
            {"model_name": list(variances.keys()), "variance": list(variances.values())}
        )

        # Sort the DataFrame by variance
        sorted_df = df.sort("variance", descending=True)

        # If top_n is specified, slice the DataFrame
        if top_n:
            sorted_df = sorted_df.head(top_n)

        if sorted_df.height == 0:
            return None

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(70, 40))

        plot_data = sorted_df.to_pandas()
        plot_data["wrapped_model_name"] = plot_data["model_name"].str.wrap(30)

        sns.barplot(
            x="wrapped_model_name",
            y="variance",
            data=plot_data,
            hue="model_name",
            palette="viridis",
            ax=ax,
            legend=False,
            dodge=False,
        )

        ax.set_ylabel("Variance of Cosine Similarities", fontsize=12)
        ax.set_title(
            "Embedding Contrast Analysis: Higher Variance is Better",
            fontsize=16,
            fontweight="bold",
        )

        ax.set_xticks(range(len(plot_data)))
        ax.set_xticklabels(
            plot_data["wrapped_model_name"],
            rotation=45,
            ha="right",
            fontsize=10,
        )

        # Add value labels on top of bars
        for i, v in enumerate(sorted_df["variance"]):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

        plt.subplots_adjust(bottom=0.25)
        plt.tight_layout(pad=3.0)
        plot_filename = os.path.join(output_dir, "embedding_variance_comparison.png")
        plt.savefig(plot_filename, bbox_inches="tight")
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

        # Create polars DataFrame
        df = pl.DataFrame(
            {
                "tsne-2d-one": tsne_results[:, 0],
                "tsne-2d-two": tsne_results[:, 1],
                "label": display_labels,
            }
        )

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 8))  # Adjusted size
        sns.scatterplot(
            x="tsne-2d-one",
            y="tsne-2d-two",
            hue="label",
            palette=label_to_color,
            data=df.to_pandas(),
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
