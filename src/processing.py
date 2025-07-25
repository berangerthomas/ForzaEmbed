import time
from typing import Callable, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    BASE_THEMES,
    CMAP,
    SIMILARITY_THRESHOLD,
)
from src.embedding_client import ProductionEmbeddingClient
from src.reporting import (
    generate_explanatory_markdown,
    generate_filtered_markdown,
    generate_heatmap_html,
)
from src.utils import chunk_text


def process_item(
    item: tuple[str, str, str, str],
    themes: list[str],
    embed_themes: np.ndarray,
    model_name: str,
    output_dir: str,
    suffix: str,
    embedding_function: Callable[[list[str]], tuple[list[list[float]], float]],
) -> tuple[str, Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    Traite un item pour générer les fichiers de sortie et retourne les embeddings et labels.

    Args:
        item (tuple): (identifiant, nom, type_lieu, texte)
        themes (list[str]): Thèmes utilisés.
        embed_themes (np.ndarray): Embeddings des thèmes.
        model_name (str): Nom du modèle.
        output_dir (str): Dossier de sortie.
        suffix (str): Suffixe pour le nom de fichier.
        embedding_function (callable): Fonction pour générer les embeddings.

    Returns:
        tuple: (message, embeddings, labels, temps de traitement)
    """
    identifiant, nom, type_lieu, texte = item
    print(f"Processing {identifiant} ({nom}) with model {model_name}...")

    if not texte or not texte.strip():
        return f"Skipped {identifiant}: empty text", None, None, 0.0

    phrases = chunk_text(texte)
    if not phrases:
        return (
            f"Skipped {identifiant}: no phrases found after chunking.",
            None,
            None,
            0.0,
        )

    embed_phrases_list, processing_time = embedding_function(phrases)

    if embed_phrases_list is None or len(embed_phrases_list) == 0:
        return (
            f"Failed to get phrase embeddings for {identifiant}.",
            None,
            None,
            processing_time,
        )

    embed_phrases = np.array(embed_phrases_list)
    if embed_phrases.size == 0:
        return (
            f"Failed to get phrase embeddings for {identifiant} (empty array).",
            None,
            None,
            processing_time,
        )
    similarites = cosine_similarity(embed_themes, embed_phrases)
    similarites_max = similarites.max(axis=0)

    # Assign each phrase to a theme based on the similarity threshold
    labels = np.argmax(similarites, axis=0)

    # Find the index for the "autre" theme, if it exists
    try:
        autre_theme_index = themes.index("autre")
        # If similarity is below threshold, classify as "autre"
        labels[similarites_max < SIMILARITY_THRESHOLD] = autre_theme_index
    except ValueError:
        # If "autre" is not in themes, just use the highest similarity
        pass

    # We clip the similarity scores to the [0, 1] range for normalization
    similarites_norm = np.clip(similarites_max, 0, 1)

    # Generate all three outputs using the new similarites_norm
    generate_heatmap_html(
        identifiant,
        nom,
        type_lieu,
        themes,
        phrases,
        similarites_norm,  # Use similarity scores for heatmap
        CMAP,
        output_dir,
        suffix,
    )
    generate_filtered_markdown(
        identifiant,
        nom,
        type_lieu,
        phrases,
        similarites_norm,  # Use similarity scores for filtering
        SIMILARITY_THRESHOLD,
        output_dir,
        suffix,
        model_name,
    )
    generate_explanatory_markdown(
        identifiant,
        nom,
        type_lieu,
        phrases,
        similarites_norm,  # Use similarity scores for explanation
        themes,
        SIMILARITY_THRESHOLD,
        output_dir,
        suffix,
        model_name,
    )

    # Return embeddings and their corresponding labels for evaluation
    return (
        f"Successfully processed {identifiant} with {model_name}.",
        embed_phrases,
        labels,
        processing_time,
    )


def run_test(
    rows: list[tuple[str, str, str, str]], model_config: dict, output_dir: str
) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    """
    Exécute les tests sur un ensemble de lignes avec la configuration du modèle.

    Args:
        rows (list[tuple[str, str, str, str]]): Liste de tuples contenant identifiants, noms,
            types de lieux, et textes.
        model_config (dict): Configuration du modèle.
        output_dir (str): Dossier de sortie.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray], float]: Un tuple contenant les embeddings de phrases, les labels et le temps de traitement total.
    """
    all_phrase_embeddings = []
    all_labels = []
    total_processing_time = 0.0
    model_type = model_config["type"]
    model_name = model_config["name"]
    print(f"\n--- Starting {model_type.upper()} Processing ({model_name}) ---")

    themes = BASE_THEMES.copy()
    suffix = f"_{model_type}_{model_name.replace('/', '_')}"

    if model_type == "local":
        model = SentenceTransformer(model_name)

        def embedding_function_local(
            texts: list[str],
        ) -> tuple[list[list[float]], float]:
            start_time = time.time()
            embeddings = model.encode(texts)
            end_time = time.time()
            return embeddings.tolist(), end_time - start_time

        embed_themes_list, _ = embedding_function_local(themes)
        embed_themes = np.array(embed_themes_list)
        print(f"Processing {len(rows)} items sequentially...")
        start_time = time.time()

        for i, item in enumerate(rows, 1):
            try:
                (
                    result,
                    embeddings,
                    labels,
                    processing_time,
                ) = process_item(
                    item,
                    themes,
                    embed_themes,
                    model_name,
                    output_dir,
                    suffix,
                    embedding_function_local,
                )
                if embeddings is not None and labels is not None:
                    all_phrase_embeddings.append(embeddings)
                    all_labels.append(labels)
                    total_processing_time += processing_time
                print(f"[{i}/{len(rows)}] {result}")
            except Exception as exc:
                print(f"[{i}/{len(rows)}] An error occurred: {exc}")

    elif model_type == "api":
        base_url = model_config["base_url"]
        client = ProductionEmbeddingClient(base_url, model_name)
        embedding_function = client.get_embeddings
        embed_themes, theme_time = embedding_function(themes)
        total_processing_time += theme_time
        if not embed_themes:
            print(
                f"❌ Could not get theme embeddings from API for {model_name}. Aborting."
            )
            return [], [], 0.0

        embed_themes = np.array(embed_themes)
        print(f"Processing {len(rows)} items sequentially via API...")
        start_time = time.time()

        for i, item in enumerate(rows, 1):
            try:
                (
                    result,
                    embeddings,
                    labels,
                    processing_time,
                ) = process_item(
                    item,
                    themes,
                    embed_themes,
                    model_name,
                    output_dir,
                    suffix,
                    embedding_function,
                )
                if embeddings is not None and labels is not None:
                    all_phrase_embeddings.append(embeddings)
                    all_labels.append(labels)
                    total_processing_time += processing_time
                print(f"[{i}/{len(rows)}] {result}")
                print("Pause de 45 secondes...")
                time.sleep(45)
            except Exception as exc:
                print(
                    f"[{i}/{len(rows)}] An error occurred while processing {item[0]}: {exc}"
                )
    else:
        print(f"Unknown model type: {model_type}")
        return [], [], 0.0

    print(
        f"--- {model_type.upper()} processing for {model_name} finished in {time.time() - start_time:.2f}s ---"
    )
    return all_phrase_embeddings, all_labels, total_processing_time
