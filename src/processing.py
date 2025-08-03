import hashlib
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
    pairwise_distances,
)
from tqdm import tqdm

from src.config import MULTIPROCESSING_CONFIG
from src.database import EmbeddingDatabase
from src.embedding_client import ProductionEmbeddingClient
from src.evaluation_metrics import (
    calculate_clustering_metrics,
    calculate_cohesion_separation,
)
from src.utils import chunk_text


def calculate_similarity(
    embed_themes: np.ndarray, embed_phrases: np.ndarray, metric: str
) -> np.ndarray:
    """Calculate similarity between theme embeddings and phrase embeddings."""
    if metric == "cosine":
        return cosine_similarity(embed_themes, embed_phrases)
    elif metric == "dot_product":
        return embed_themes @ embed_phrases.T
    elif metric == "euclidean":
        distances = euclidean_distances(embed_themes, embed_phrases)
        return 1 / (1 + distances)
    elif metric == "manhattan":
        distances = manhattan_distances(embed_themes, embed_phrases)
        return 1 / (1 + distances)
    elif metric == "chebyshev":
        distances = pairwise_distances(embed_themes, embed_phrases, metric="chebyshev")
        return 1 / (1 + distances)
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")


def get_text_hash(text: str) -> str:
    """Generates a SHA-256 hash for a given text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get_or_create_embeddings(
    db: EmbeddingDatabase,
    embedding_function: Callable,
    base_model_name: str,
    phrases: List[str],
) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Retrieves embeddings from the SQLite cache or generates and caches them if they don't exist.
    """
    phrase_hashes = {phrase: get_text_hash(phrase) for phrase in phrases}

    # Check for existing embeddings in the cache WITH model context
    existing_embeddings = db.get_embeddings_by_hashes(
        base_model_name, list(phrase_hashes.values())
    )

    # Identify which phrases need to be embedded
    phrases_to_embed = [
        phrase for phrase, h in phrase_hashes.items() if h not in existing_embeddings
    ]

    total_processing_time = 0.0
    if phrases_to_embed:
        # Generate embeddings for new phrases
        new_embeddings_list, processing_time = embedding_function(phrases_to_embed)
        total_processing_time = processing_time

        if new_embeddings_list:
            # Create a map of hash to new embedding vector
            new_embeddings_map = {
                phrase_hashes[phrase]: np.array(embedding)
                for phrase, embedding in zip(phrases_to_embed, new_embeddings_list)
            }

            # Save the new embeddings to the cache WITH model context
            db.save_embeddings_batch(base_model_name, new_embeddings_map)

            # Add the new embeddings to the map of existing embeddings for this run
            existing_embeddings.update(new_embeddings_map)

    # Combine all embeddings (cached and new) for the current set of phrases
    all_embeddings_for_phrases = {
        h: existing_embeddings[h]
        for p, h in phrase_hashes.items()
        if h in existing_embeddings
    }

    return all_embeddings_for_phrases, total_processing_time


def run_test(
    rows: List[Tuple[str, str, str, str]],
    db: EmbeddingDatabase,
    model_config: Dict[str, Any],
    chunk_size: int,
    chunk_overlap: int,
    themes: List[str],
    theme_name: str,
    chunking_strategy: str,
    similarity_metric: str,
    processed_files: List[str],
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Processes a test run using SQLite for metadata and embeddings cache.
    """
    model_type = model_config["type"]
    model_name = model_config["name"]
    results = {"files": {}}

    embedding_function: Callable
    if model_type in ["sentence_transformers", "fastembed", "huggingface"]:

        def get_embeddings(texts):
            return model_config["function"](
                texts,
                model_name=model_name,
                expected_dimension=model_config.get("dimensions"),
            )

        embedding_function = get_embeddings
    else:
        # Determine appropriate batch size based on API provider
        api_batch_sizes = MULTIPROCESSING_CONFIG.get("api_batch_sizes", {})
        model_lower = model_name.lower()

        initial_batch_size = api_batch_sizes.get("default", 100)
        for provider, batch_size in api_batch_sizes.items():
            if provider in model_lower:
                initial_batch_size = batch_size
                break

        client = ProductionEmbeddingClient(
            model_config["base_url"],
            model_name,
            expected_dimension=model_config.get("dimensions"),
            timeout=model_config.get("timeout", 30),
        )
        # Store initial batch size for use in subdivision
        client._initial_batch_size = initial_batch_size
        embedding_function = client.get_embeddings

    # Cache themes embeddings too WITH model context
    themes_embeddings_map, _ = _get_or_create_embeddings(
        db, embedding_function, model_config["name"], themes
    )

    if not themes_embeddings_map:
        raise RuntimeError(
            f"Embedding for themes failed for model '{model_name}'. "
            "Check API response or local model configuration."
        )

    # Reconstruct themes embeddings in the same order
    theme_hashes = [get_text_hash(theme) for theme in themes]
    embed_themes_list = [
        themes_embeddings_map[h] for h in theme_hashes if h in themes_embeddings_map
    ]
    embed_themes = np.array(embed_themes_list)

    unprocessed_rows = [row for row in rows if row[0] not in processed_files]

    for item in tqdm(
        unprocessed_rows,
        desc=f"Processing {model_name}",
        leave=False,
        disable=not show_progress,
    ):
        identifiant, _, _, texte = item
        if not texte or not texte.strip():
            continue

        item_phrases = chunk_text(texte, chunk_size, chunk_overlap, chunking_strategy)
        if not item_phrases:
            continue

        all_embeddings_map, p_time = _get_or_create_embeddings(
            db,
            embedding_function,
            model_config["name"],
            item_phrases,
        )

        phrase_hashes = {p: get_text_hash(p) for p in item_phrases}
        item_embed_phrases = np.array(
            [
                all_embeddings_map[h]
                for p in item_phrases
                if (h := phrase_hashes[p]) in all_embeddings_map
            ]
        )

        # Skip if phrase embeddings are empty (API error)
        if item_embed_phrases.size == 0:
            # Skip this file if embedding failed, do not write to results
            continue

        # --- Vérification stricte des dimensions ---
        if embed_themes.shape[1] != item_embed_phrases.shape[1]:
            raise ValueError(
                f"Dimension mismatch for file '{identifiant}': "
                f"theme embeddings dim={embed_themes.shape[1]}, "
                f"phrase embeddings dim={item_embed_phrases.shape[1]} "
                f"(model: {model_name}). "
                "This indicates that themes and phrases were not embedded with the same model or configuration."
            )

        similarites = calculate_similarity(
            embed_themes, item_embed_phrases, similarity_metric
        )
        labels = np.argmax(similarites, axis=0)

        cohesion_sep = calculate_cohesion_separation(item_embed_phrases, labels)
        clustering_metrics = calculate_clustering_metrics(item_embed_phrases, labels)

        results["files"][identifiant] = {
            "phrases": item_phrases,
            "similarities": similarites.max(axis=0),
            "metrics": {
                **cohesion_sep,
                **clustering_metrics,
                "processing_time": p_time,
                "mean_similarity": float(np.mean(similarites.max(axis=0))),
            },
        }

    return {"results": results}
