import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
from dotenv import load_dotenv
from evaluation_metrics import (
    calculate_centroid_distance,
    calculate_classification_metrics,
    calculate_separation_score,
    plot_centroid_distances,
    plot_roc_curves,
    plot_separation_scores,
    plot_tsne_projection,
)
from matplotlib.colors import LinearSegmentedColormap
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
THRESHOLD = 0.6  # Similarity threshold for relevance
CONTEXT_SIZE = 2  # Number of sentences to retrieve around similar zones (0 = only the sentence, 1 = sentence +/- 1 sentence)
REGEX_BONUS = 0  # Bonus added to the similarity score if regex patterns are found


# --- Theme and Keyword Generation ---
def generate_themes_and_keywords():
    """
    Generates a list of themes for semantic search and a structured dictionary of keywords for regex.
    """
    jours = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    actions = ["ouvert", "fermé", "ouverture", "fermeture", "accueil"]
    concepts_generaux = [
        "horaires d'ouverture",
        "heures d'ouverture",
        "accueil du public",
        "jour de fermeture",
        "horaires et jours de la semaine",
        "périodes d'ouverture",
        "période scolaire",
        "vacances scolaires",
        "jours fériés",
        "jours spéciaux",
        "horaires spéciaux",
    ]
    expressions_temporelles = [
        "le matin",
        "l'après-midi",
        "de 9h à 12h",
        "de 14h à 18h",
        "de 10h00 à 19h00",
    ]

    # Generate themes by combining actions and days
    themes_jours = [f"{action} le {jour}" for action in actions[:2] for jour in jours]

    # Combine all themes into a single list
    base_themes = concepts_generaux + expressions_temporelles + themes_jours

    # Create a dictionary of keywords for the regex
    keywords = {
        "jours": jours,
        "actions": actions,
    }
    return base_themes, keywords


BASE_THEMES, KEYWORDS_FOR_REGEX = generate_themes_and_keywords()

# --- Model Configuration ---
MODELS_TO_TEST = [
    {
        "type": "local",
        "name": "all-mpnet-base-v2",
        "function": lambda model_name: SentenceTransformer(model_name).encode,
    },
    {
        "type": "local",
        "name": "multi-qa-mpnet-base-dot-v1",
        "function": lambda model_name: SentenceTransformer(model_name).encode,
    },
    {
        "type": "local",
        "name": "all-distilroberta-v1",
        "function": lambda model_name: SentenceTransformer(model_name).encode,
    },
    {
        "type": "local",
        "name": "all-MiniLM-L12-v2",
        "function": lambda model_name: SentenceTransformer(model_name).encode,
    },
    {
        "type": "local",
        "name": "multi-qa-distilbert-cos-v1",
        "function": lambda model_name: SentenceTransformer(model_name).encode,
    },
    {
        "type": "local",
        "name": "all-MiniLM-L6-v2",
        "function": lambda model_name: SentenceTransformer(model_name).encode,
    },
    {
        "type": "local",
        "name": "multi-qa-MiniLM-L6-cos-v1",
        "function": lambda model_name: SentenceTransformer(model_name).encode,
    },
    {
        "type": "local",
        "name": "paraphrase-multilingual-mpnet-base-v2",
        "function": lambda model_name: SentenceTransformer(model_name).encode,
    },
    {
        "type": "local",
        "name": "paraphrase-albert-small-v2",
        "function": lambda model_name: SentenceTransformer(model_name).encode,
    },
    {
        "type": "local",
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "function": lambda model_name: SentenceTransformer(model_name).encode,
    },
    {
        "type": "local",
        "name": "paraphrase-MiniLM-L3-v2",
        "function": lambda model_name: SentenceTransformer(model_name).encode,
    },
    {
        "type": "local",
        "name": "distiluse-base-multilingual-cased-v1",
        "function": lambda model_name: SentenceTransformer(model_name).encode,
    },
    {
        "type": "local",
        "name": "distiluse-base-multilingual-cased-v2",
        "function": lambda model_name: SentenceTransformer(model_name).encode,
    },
    {
        "type": "api",
        "name": "nomic-embed-text",
        "base_url": "https://api.erasme.homes/v1",
        "function": lambda client: client.get_embeddings,
    },
]
# MODELS_TO_TEST = [
#     {
#         "type": "local",
#         "name": "paraphrase-multilingual-mpnet-base-v2",
#         "function": lambda model_name: SentenceTransformer(model_name).encode,
#     },
#     {
#         "type": "local",
#         "name": "paraphrase-multilingual-MiniLM-L12-v2",
#         "function": lambda model_name: SentenceTransformer(model_name).encode,
#     },
#     {
#         "type": "local",
#         "name": "distiluse-base-multilingual-cased-v1",
#         "function": lambda model_name: SentenceTransformer(model_name).encode,
#     },
#     {
#         "type": "local",
#         "name": "distiluse-base-multilingual-cased-v2",
#         "function": lambda model_name: SentenceTransformer(model_name).encode,
#     },
#     {
#         "type": "api",
#         "name": "nomic-embed-text",
#         "base_url": "https://api.erasme.homes/v1",
#         "function": lambda client: client.get_embeddings,
#     },
# ]

# API-based model (Ollama compatible)
BASE_URL = "https://api.erasme.homes/v1"  # Default, can be overridden
EMBEDDING_MODEL = "nomic-embed-text"  # Default, can be overridden

# --- Output Configuration ---
# Définit le répertoire de sortie dans le même dossier que le script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "heatmaps")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Colormap for Heatmaps ---
CMAP = LinearSegmentedColormap.from_list(
    "custom",
    ["#2b83ba", "#abdda4", "#ffffbf", "#fdae61", "#d7191c"],
)

# --- Helper Functions ---


def chunk_text(text: str) -> List[str]:
    """
    Splits text into sentences using regex.
    Also handles simple list-like structures by splitting on newlines.
    """
    # First, split by sentence-ending punctuation.
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Then, further split each "sentence" by newlines to handle lists
    final_chunks = []
    for sentence in sentences:
        lines = sentence.split("\n")
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:  # Only add non-empty lines
                final_chunks.append(stripped_line)

    return final_chunks


def contains_horaire_pattern(text: str, keywords: dict) -> bool:
    """
    Checks if a text string contains patterns indicative of opening hours,
    using a dynamically generated regex from keywords.
    """
    # Build regex patterns from the keywords dictionary
    time_pattern = r"\d{1,2}h(\d{2})?"
    days_pattern = r"\b(" + "|".join(keywords["jours"]) + r")\b"
    keyword_pattern = r"\b(" + "|".join(keywords["actions"]) + r")\b"
    range_pattern = r"\d{1,2}h(\d{2})?\s*[-\/]\s*\d{1,2}h(\d{2})?"

    # Combine all patterns into a single regex for efficiency
    combined_pattern = "|".join(
        [time_pattern, days_pattern, keyword_pattern, range_pattern]
    )

    # Check if the combined pattern is found
    if re.search(combined_pattern, text, re.IGNORECASE):
        return True

    return False


def extract_context_around_phrase(phrases, phrase_index, context_window=2):
    """Extracts context around a specific phrase index and highlights the target phrase."""
    start_idx = max(0, phrase_index - context_window)
    end_idx = min(len(phrases), phrase_index + context_window + 1)
    context_phrases = phrases[start_idx:end_idx]

    context_with_highlight = []
    for i, phrase in enumerate(context_phrases):
        actual_idx = start_idx + i
        if actual_idx == phrase_index:
            context_with_highlight.append(f"**{phrase.strip()}**")
        else:
            context_with_highlight.append(phrase.strip())
    return " ".join(context_with_highlight)


# --- Output Generation Functions ---


def generate_heatmap_html(
    identifiant,
    nom,
    type_lieu,
    themes,
    phrases,
    similarites_norm,
    cmap,
    output_dir,
    suffix,
):
    """Generates and saves an HTML file with colored phrases based on similarity."""
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


def generate_filtered_markdown(
    identifiant,
    nom,
    type_lieu,
    phrases,
    similarites_norm,
    threshold,
    context_size,
    output_dir,
    suffix,
    model_name,
):
    """Generates a simple markdown file with only the relevant phrases."""
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


def analyze_and_visualize_variance(model_embeddings, output_dir):
    """
    Analyzes the variance of embeddings for each model and generates a visualization
    to identify the model with the highest discriminant power.

    Args:
        model_embeddings (dict): A dictionary where keys are model names and
                                 values are lists of embedding arrays.
        output_dir (str): The directory to save the visualization.
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
    best_model = max(variances, key=variances.get)
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


def generate_explanatory_markdown(
    identifiant,
    nom,
    type_lieu,
    phrases,
    similarites_norm,
    themes,
    threshold,
    context_window,
    output_dir,
    suffix,
    model_name,
):
    """Generates a detailed markdown summary of high-similarity zones."""
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


# --- API Client for Production Embeddings ---
load_dotenv()


class ProductionEmbeddingClient:
    """Simplified client to get embeddings from an Ollama-compatible API."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        api_key = os.environ.get("EMBED_API_KEY_OPENAI")
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model, "input": texts}
        try:
            response = self.session.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return [data["embedding"] for data in result["data"]]
        except requests.exceptions.RequestException as e:
            print(f"❌ API Error: {e}")
            return []
        except (KeyError, IndexError) as e:
            print(f"❌ API Response Parsing Error: {e}")
            return []


# --- Processing Functions ---


def process_item(
    item, themes, embed_themes, model_name, output_dir, suffix, embedding_function
):
    """Generic function to process a single item."""
    identifiant, nom, type_lieu, texte = item
    print(f"Processing {identifiant} ({nom}) with model {model_name}...")

    if not texte or not texte.strip():
        return f"Skipped {identifiant}: empty text", None

    phrases = chunk_text(texte)
    if not phrases:
        return f"Skipped {identifiant}: no phrases found after chunking.", None

    embed_phrases_list = embedding_function(phrases)

    if embed_phrases_list is None or len(embed_phrases_list) == 0:
        return f"Failed to get phrase embeddings for {identifiant}.", None

    embed_phrases = np.array(embed_phrases_list)
    if embed_phrases.size == 0:
        return f"Failed to get phrase embeddings for {identifiant} (empty array).", None
    similarites = cosine_similarity(embed_themes, embed_phrases)
    similarites_max = similarites.max(axis=0)

    # We clip the similarity scores to the [0, 1] range for normalization
    similarites_norm = np.clip(similarites_max, 0, 1)

    # --- Nuanced Regex + Semantic Scoring ---
    # We give a bonus to phrases that contain regex patterns, instead of zeroing them out.
    # This preserves the semantic score while rewarding strong indicators.
    bonus_scores = np.array(
        [
            REGEX_BONUS if contains_horaire_pattern(p, KEYWORDS_FOR_REGEX) else 0
            for p in phrases
        ]
    )
    combined_scores = similarites_norm + bonus_scores

    # We clip the final scores to ensure they stay within the [0, 1] range
    final_scores = np.clip(combined_scores, 0, 1)

    # Generate labels for evaluation metrics (ground truth simulation)
    labels = [contains_horaire_pattern(p, KEYWORDS_FOR_REGEX) for p in phrases]

    # Generate all three outputs using the new final_scores
    generate_heatmap_html(
        identifiant,
        nom,
        type_lieu,
        themes,
        phrases,
        final_scores,  # Use final scores for heatmap
        CMAP,
        output_dir,
        suffix,
    )
    generate_filtered_markdown(
        identifiant,
        nom,
        type_lieu,
        phrases,
        final_scores,  # Use final scores for filtering
        THRESHOLD,
        CONTEXT_SIZE,
        output_dir,
        suffix,
        model_name,
    )
    generate_explanatory_markdown(
        identifiant,
        nom,
        type_lieu,
        phrases,
        final_scores,  # Use final scores for explanation
        themes,
        THRESHOLD,
        CONTEXT_SIZE,
        output_dir,
        suffix,
        model_name,
    )

    # Return embeddings and their corresponding labels for evaluation
    return (
        f"Successfully processed {identifiant} with {model_name}.",
        embed_phrases,
        labels,
        final_scores,
    )


# --- Main Execution Logic ---


def run_test(rows, model_config):
    """Processes all items based on the provided model configuration."""
    all_phrase_embeddings = []
    all_labels = []
    all_scores = []
    model_type = model_config["type"]
    model_name = model_config["name"]
    print(f"\n--- Starting {model_type.upper()} Processing ({model_name}) ---")

    themes = BASE_THEMES.copy()
    suffix = f"_{model_type}_{model_name.replace('/', '_')}"

    if model_type == "local":
        model = SentenceTransformer(model_name)
        embedding_function = model.encode
        embed_themes = embedding_function(themes)
        print(f"Processing {len(rows)} items sequentially...")
        start_time = time.time()

        for i, item in enumerate(rows, 1):
            try:
                result, embeddings, labels, scores = process_item(
                    item,
                    themes,
                    embed_themes,
                    model_name,
                    OUTPUT_DIR,
                    suffix,
                    embedding_function,
                )
                if embeddings is not None:
                    all_phrase_embeddings.append(embeddings)
                    all_labels.extend(labels)
                    all_scores.extend(scores)
                print(f"[{i}/{len(rows)}] {result}")
            except Exception as exc:
                print(f"[{i}/{len(rows)}] An error occurred: {exc}")

    elif model_type == "api":
        base_url = model_config["base_url"]
        client = ProductionEmbeddingClient(base_url, model_name)
        embedding_function = client.get_embeddings
        embed_themes = embedding_function(themes)
        if not embed_themes:
            print(
                f"❌ Could not get theme embeddings from API for {model_name}. Aborting."
            )
            return []

        embed_themes = np.array(embed_themes)
        print(f"Processing {len(rows)} items sequentially via API...")
        start_time = time.time()

        for i, item in enumerate(rows, 1):
            try:
                result, embeddings, labels, scores = process_item(
                    item,
                    themes,
                    embed_themes,
                    model_name,
                    OUTPUT_DIR,
                    suffix,
                    embedding_function,
                )
                if embeddings is not None:
                    all_phrase_embeddings.append(embeddings)
                    all_labels.extend(labels)
                    all_scores.extend(scores)
                print(f"[{i}/{len(rows)}] {result}")
            except Exception as exc:
                print(
                    f"[{i}/{len(rows)}] An error occurred while processing {item[0]}: {exc}"
                )
    else:
        print(f"Unknown model type: {model_type}")
        return []

    print(
        f"--- {model_type.upper()} processing for {model_name} finished in {time.time() - start_time:.2f}s ---"
    )
    return all_phrase_embeddings, all_labels, all_scores


if __name__ == "__main__":
    # --- Data Loading ---
    conn = sqlite3.connect("data/SmartWatch.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT l.identifiant, l.nom, l.type_lieu, r.markdown_nettoye FROM resultats_extraction r JOIN lieux l ON r.lieu_id = l.identifiant"
    )
    all_rows = cursor.fetchall()
    conn.close()

    # Filter for specific items for testing
    identifiants_to_test = [
        "S3391",
        "S3736",
        "S6914",
    ]
    test_rows = [row for row in all_rows if row[0] in identifiants_to_test]
    test_rows = all_rows  # Uncomment this line to process all items

    # --- Run Processing ---
    model_embeddings_for_variance = {}
    evaluation_results = {}

    for config in MODELS_TO_TEST:
        model_name = config["name"]
        # Each run_test returns embeddings, labels, and scores for one model
        embeddings_list, labels, scores = run_test(test_rows, config)

        if embeddings_list:
            # --- Store embeddings for original variance analysis ---
            model_embeddings_for_variance[model_name] = embeddings_list

            # --- Prepare data for new evaluation metrics ---
            # Flatten the list of embedding arrays into a single array
            all_embeddings = np.vstack(
                [emb for emb in embeddings_list if emb is not None and emb.size > 0]
            )
            # Ensure labels and scores are numpy arrays
            labels = np.array(labels)
            scores = np.array(scores)

            if all_embeddings.shape[0] != len(labels):
                print(
                    f"Mismatch between embeddings ({all_embeddings.shape[0]}) and labels ({len(labels)}) for {model_name}. Skipping evaluation."
                )
                continue

            # Separate embeddings based on ground truth labels
            embeddings_pertinents = all_embeddings[labels == True]
            embeddings_non_pertinents = all_embeddings[labels == False]

            # --- Calculate all new metrics ---
            print(f"\n--- Calculating Evaluation Metrics for {model_name} ---")
            evaluation_results[model_name] = {}

            # 1. Separation Score
            sep_score, sim_intra, sim_inter = calculate_separation_score(
                embeddings_pertinents, embeddings_non_pertinents
            )
            evaluation_results[model_name]["separation_score"] = sep_score
            evaluation_results[model_name]["sim_intra"] = sim_intra
            evaluation_results[model_name]["sim_inter"] = sim_inter
            print(f"  - Separation Score: {sep_score:.3f}")

            # 2. Classification Metrics (ROC/AUC)
            class_metrics = calculate_classification_metrics(labels, scores)
            evaluation_results[model_name].update(class_metrics)
            print(f"  - ROC AUC: {class_metrics['roc_auc']:.3f}")

            # 3. Centroid Distance
            dist = calculate_centroid_distance(
                embeddings_pertinents, embeddings_non_pertinents
            )
            evaluation_results[model_name]["centroid_distance"] = dist
            print(f"  - Centroid Distance: {dist:.3f}")

            # 4. t-SNE Projection
            print("  - Generating t-SNE plot...")
            plot_tsne_projection(all_embeddings, labels, model_name, OUTPUT_DIR)

    # --- Generate Comparison Plots ---
    print("\n--- Generating Final Comparison Plots ---")
    if evaluation_results:
        plot_separation_scores(evaluation_results, OUTPUT_DIR)
        plot_roc_curves(evaluation_results, OUTPUT_DIR)
        plot_centroid_distances(evaluation_results, OUTPUT_DIR)
        print("  - Separation, ROC, and Centroid plots generated.")
    else:
        print("No evaluation results to plot.")

    # After all models have been processed, analyze the collected embeddings for variance
    if model_embeddings_for_variance:
        analyze_and_visualize_variance(model_embeddings_for_variance, OUTPUT_DIR)
    else:
        print("\nNo embeddings were generated. Skipping variance analysis.")

    print(f"\n✅ All processing complete. Outputs are in the '{OUTPUT_DIR}' directory.")
