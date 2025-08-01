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

from src.config import (
    SIMILARITY_THRESHOLD,
)
from src.database import EmbeddingDatabase
from src.embedding_client import ProductionEmbeddingClient
from src.evaluation_metrics import (
    calculate_clustering_metrics,
    calculate_cohesion_separation,
)
from src.utils import chunk_text, to_python_type


def calculate_similarity(
    embed_themes: np.ndarray, embed_phrases: np.ndarray, metric: str
) -> np.ndarray:
    """Calculate similarity between theme embeddings and phrase embeddings."""
    if metric == "cosine":
        return cosine_similarity(embed_themes, embed_phrases)
    elif metric == "dot_product":
        # Note: Not normalized, magnitude matters.
        return embed_themes @ embed_phrases.T
    elif metric == "euclidean":
        # Invert and normalize so that higher is better
        distances = euclidean_distances(embed_themes, embed_phrases)
        return 1 / (1 + distances)
    elif metric == "manhattan":
        # Invert and normalize
        distances = manhattan_distances(embed_themes, embed_phrases)
        return 1 / (1 + distances)
    elif metric == "chebyshev":
        # Invert and normalize
        distances = pairwise_distances(embed_themes, embed_phrases, metric="chebyshev")
        return 1 / (1 + distances)
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")


def get_text_hash(text: str) -> str:
    """Generates a SHA-256 hash for a given text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def process_item(
    item: Tuple[str, str, str, str],
    themes: List[str],
    embed_themes: np.ndarray,
    phrases: List[str],
    embed_phrases: np.ndarray,
    processing_time: float,
    similarity_metric: str,
) -> Tuple[str, Any]:
    """
    Process an item with pre-computed embeddings and return a result dictionary.
    """
    identifiant, _, _, _ = item

    similarites = calculate_similarity(embed_themes, embed_phrases, similarity_metric)
    similarites_max = similarites.max(axis=0)
    similarites_norm = np.clip(similarites_max, 0, 1)

    labels = np.argmax(similarites, axis=0)
    try:
        autre_theme_index = themes.index("autre")
        labels[similarites_max < SIMILARITY_THRESHOLD] = autre_theme_index
    except ValueError:
        pass

    # Static reports are now generated at the end via --generate-reports

    # Compute metrics for this specific file
    cohesion_sep = calculate_cohesion_separation(embed_phrases, labels)
    clustering_metrics = calculate_clustering_metrics(embed_phrases, labels)

    metrics = {
        **cohesion_sep,
        **clustering_metrics,
        "processing_time": processing_time,
        "mean_similarity": float(np.mean(similarites_norm)),
    }

    result_data = {
        "phrases": phrases,
        "themes": themes,
        "similarities": similarites_norm,
        "metrics": metrics,
        "embeddings_data": {"embeddings": embed_phrases, "labels": labels},
    }
    return f"Successfully processed {identifiant}", to_python_type(result_data)


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
    Enhanced run_test with better memory management and batch processing.
    """
    model_type = model_config["type"]
    model_name = model_config["name"]
    results = {"files": {}}

    # --- 1. Setup embedding function ---
    embedding_function: Callable
    if model_type in ["sentence_transformers", "fastembed", "huggingface"]:
        embedding_function = lambda texts: model_config["function"](
            texts,
            model_name=model_name,
            expected_dimension=model_config.get("dimensions"),
        )
    elif model_type == "api":
        client = ProductionEmbeddingClient(
            model_config["base_url"],
            model_name,
            expected_dimension=model_config.get("dimensions"),
            timeout=model_config.get("timeout", 30),
        )
        embedding_function = client.get_embeddings
    else:
        tqdm.write(f"Unknown model type: {model_type}")
        return results

    # --- 2. Embed themes (never cached) ---
    embed_themes_list, _ = embedding_function(themes)
    if not embed_themes_list:
        tqdm.write(f"❌ Could not get theme embeddings for {model_name}. Aborting.")
        return results
    embed_themes = np.array(embed_themes_list)

    # --- 3. Filter and batch process phrases more efficiently ---
    unprocessed_rows = [row for row in rows if row[0] not in processed_files]

    # Process in smaller batches to reduce memory usage
    batch_size = min(50, len(unprocessed_rows))  # Adjust based on available memory

    all_phrases_map = {}
    unique_phrases = set()

    for batch_start in range(0, len(unprocessed_rows), batch_size):
        batch_end = min(batch_start + batch_size, len(unprocessed_rows))
        batch_rows = unprocessed_rows[batch_start:batch_end]

        for item in batch_rows:
            identifiant, _, _, texte = item
            if not texte or not texte.strip():
                continue
            phrases = chunk_text(texte, chunk_size, chunk_overlap, chunking_strategy)
            if phrases:
                all_phrases_map[identifiant] = phrases
                unique_phrases.update(phrases)

    # --- 4. Optimized phrase embedding with batch processing ---
    unique_phrases_list = list(unique_phrases)
    phrase_hashes = {phrase: get_text_hash(phrase) for phrase in unique_phrases_list}
    hashes_to_check = list(phrase_hashes.values())

    # Batch database lookups
    cached_embeddings = db.get_cached_embeddings(model_name, hashes_to_check)

    phrases_to_embed = [
        phrase
        for phrase in unique_phrases_list
        if phrase_hashes[phrase] not in cached_embeddings
    ]

    total_processing_time = 0.0
    newly_embedded_map = {}

    if phrases_to_embed:
        # Process embeddings in batches for API efficiency
        embedding_batch_size = 100 if model_type == "api" else 500

        for i in range(0, len(phrases_to_embed), embedding_batch_size):
            batch_phrases = phrases_to_embed[i : i + embedding_batch_size]

            if not show_progress:
                tqdm.write(
                    f"   Embedding batch {i // embedding_batch_size + 1}/{(len(phrases_to_embed) + embedding_batch_size - 1) // embedding_batch_size} ({len(batch_phrases)} phrases)"
                )

            new_embeddings, processing_time = embedding_function(batch_phrases)
            total_processing_time += processing_time

            if new_embeddings:
                batch_embedded_map = {
                    phrase: embedding
                    for phrase, embedding in zip(batch_phrases, new_embeddings)
                }
                newly_embedded_map.update(batch_embedded_map)

                # Cache embeddings immediately to prevent loss
                hashes_to_cache = {
                    phrase_hashes[phrase]: embedding
                    for phrase, embedding in batch_embedded_map.items()
                }
                db.cache_embeddings(model_name, hashes_to_cache)
            else:
                tqdm.write(f"⚠️ Failed to embed batch for {model_name}.")

    # --- 5. Combine cached and new embeddings ---
    final_embeddings_map = {}
    for phrase in unique_phrases_list:
        text_hash = phrase_hashes[phrase]
        if text_hash in cached_embeddings:
            final_embeddings_map[phrase] = cached_embeddings[text_hash]
        elif phrase in newly_embedded_map:
            final_embeddings_map[phrase] = newly_embedded_map[phrase]

    # --- 6. Process items with memory-efficient iteration ---
    iterable = (
        tqdm(
            unprocessed_rows,
            desc=f"Processing files ({model_name})",
            unit="file",
            initial=len(processed_files),
            total=len(rows),
            leave=False,  # Don't leave progress bar after completion
        )
        if show_progress
        else unprocessed_rows
    )

    for item in iterable:
        identifiant = item[0]
        if identifiant not in all_phrases_map:
            continue

        item_phrases = all_phrases_map[identifiant]
        item_embed_phrases_list = [
            final_embeddings_map[p] for p in item_phrases if p in final_embeddings_map
        ]

        if not item_embed_phrases_list:
            if show_progress:
                tqdm.write(
                    f"Skipping {identifiant}: could not find embeddings for its phrases."
                )
            continue

        item_embed_phrases = np.array(item_embed_phrases_list)

        try:
            message, file_data = process_item(
                item,
                themes,
                embed_themes,
                item_phrases,
                item_embed_phrases,
                total_processing_time / len(unique_phrases) if unique_phrases else 0,
                similarity_metric,
            )
            if file_data:
                results["files"][identifiant] = file_data
        except Exception as e:
            if show_progress:
                tqdm.write(f"Error processing {identifiant}: {e}")

    return results
