import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    CMAP,
    SIMILARITY_THRESHOLD,
)
from src.embedding_client import ProductionEmbeddingClient
from src.evaluation_metrics import (
    calculate_clustering_metrics,
    calculate_cohesion_separation,
)
from src.reporting import (
    generate_explanatory_markdown,
    generate_filtered_markdown,
    generate_heatmap_html,
)
from src.utils import chunk_text


def process_item(
    item: Tuple[str, str, str, str],
    themes: List[str],
    embed_themes: np.ndarray,
    model_name: str,
    embedding_function: Callable[
        [List[str]], Tuple[Optional[List[List[float]]], float]
    ],
    output_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    theme_name: str,
    chunking_strategy: str,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Traite un item, génère les rapports statiques, et retourne un dictionnaire de résultats.
    """
    identifiant, nom, type_lieu, texte = item
    print(f"Processing {identifiant} ({nom}) with model {model_name}...")

    if not texte or not texte.strip():
        return f"Skipped {identifiant}: empty text", None

    phrases = chunk_text(texte, chunk_size, chunk_overlap, chunking_strategy)
    if not phrases:
        return f"Skipped {identifiant}: no phrases after chunking.", None

    embed_phrases_list, processing_time = embedding_function(phrases)
    if not embed_phrases_list:
        return f"Failed to get phrase embeddings for {identifiant}.", None

    embed_phrases = np.array(embed_phrases_list)
    similarites = cosine_similarity(embed_themes, embed_phrases)
    similarites_max = similarites.max(axis=0)
    similarites_norm = np.clip(similarites_max, 0, 1)

    labels = np.argmax(similarites, axis=0)
    try:
        autre_theme_index = themes.index("autre")
        labels[similarites_max < SIMILARITY_THRESHOLD] = autre_theme_index
    except ValueError:
        pass

    # Les rapports statiques sont maintenant générés à la fin via --generate-reports

    # Calculer les métriques pour ce fichier spécifique
    cohesion_sep = calculate_cohesion_separation(embed_phrases, labels)
    clustering_metrics = calculate_clustering_metrics(embed_phrases, labels)

    metrics = {
        **cohesion_sep,
        **clustering_metrics,
        "processing_time": processing_time,
        "mean_similarity": np.mean(similarites_norm),
    }

    return f"Successfully processed {identifiant}", {
        "phrases": phrases,
        "themes": themes,
        "similarities": similarites_norm.tolist(),
        "metrics": metrics,
        "embeddings_data": {
            "embeddings": embed_phrases,
            "labels": labels,
        },
    }


def run_test(
    rows: List[Tuple[str, str, str, str]],
    model_config: Dict[str, Any],
    chunk_size: int,
    chunk_overlap: int,
    themes: List[str],
    output_dir: str,
    theme_name: str,
    chunking_strategy: str,
) -> Dict[str, Any]:
    """
    Exécute les tests et retourne les données structurées pour la page web.
    """
    model_type = model_config["type"]
    model_name = model_config["name"]
    run_name = f"{model_name}_cs{chunk_size}_co{chunk_overlap}_t{theme_name}_s{chunking_strategy}"
    print(f"\n--- Starting {model_type.upper()} Processing ({run_name}) ---")

    results = {"files": {}}

    embedding_function: Callable
    if model_type == "local":
        # Utilise le client local qui gère les modèles en singletons
        embedding_function = lambda texts: model_config["function"](
            texts, model_name=model_name
        )
    elif model_type == "api":
        client = ProductionEmbeddingClient(model_config["base_url"], model_name)
        embedding_function = client.get_embeddings
    else:
        print(f"Unknown model type: {model_type}")
        return results

    embed_themes_list, _ = embedding_function(themes)
    if not embed_themes_list:
        print(f"❌ Could not get theme embeddings for {model_name}. Aborting.")
        return results
    embed_themes = np.array(embed_themes_list)

    for i, item in enumerate(rows, 1):
        identifiant = item[0]
        try:
            message, file_data = process_item(
                item,
                themes,
                embed_themes,
                model_name,
                embedding_function,
                output_dir,
                chunk_size,
                chunk_overlap,
                theme_name,
                chunking_strategy,
            )
            if file_data:
                results["files"][identifiant] = file_data
            print(f"[{i}/{len(rows)}] {message}")
            if model_type == "api":
                time.sleep(60)
        except Exception as e:
            print(f"[{i}/{len(rows)}] Error processing {identifiant}: {e}")

    return results
