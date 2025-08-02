import hashlib
import json
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pyarrow as pa
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
    pairwise_distances,
)
from tqdm import tqdm

from src.config import SIMILARITY_THRESHOLD
from src.database import EmbeddingDatabase
from src.embedding_client import ProductionEmbeddingClient
from src.lancedb_client import LanceDBClient
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
    lance_db: LanceDBClient,
    embedding_function: Callable,
    phrases: List[str],
    model_name: str,
    model_dimensions: int,
) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Retrieves embeddings from LanceDB or generates and caches them if they don't exist.
    """
    table_name = f"embed_{model_name.replace('-', '_').replace('/', '_')}"
    phrase_hashes = {phrase: get_text_hash(phrase) for phrase in phrases}

    schema = pa.schema(
        [
            pa.field("text_hash", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), list_size=model_dimensions)),
        ]
    )
    lance_db.create_table_if_not_exists(table_name, schema)

    existing_embeddings = {}
    try:
        table = lance_db.db.open_table(table_name)
        all_records = table.to_pandas()
        if not all_records.empty:
            existing_hashes = set(all_records["text_hash"])
            phrases_to_check = [
                h for h in phrase_hashes.values() if h in existing_hashes
            ]
            if phrases_to_check:
                results = all_records[
                    all_records["text_hash"].isin(phrases_to_check)
                ]
                for _, row in results.iterrows():
                    existing_embeddings[row["text_hash"]] = np.array(row["vector"])
    except FileNotFoundError:
        pass

    phrases_to_embed = [
        phrase for phrase, h in phrase_hashes.items() if h not in existing_embeddings
    ]

    total_processing_time = 0.0
    if phrases_to_embed:
        new_embeddings, processing_time = embedding_function(phrases_to_embed)
        total_processing_time = processing_time
        if new_embeddings:
            new_data = [
                {"text_hash": phrase_hashes[phrase], "vector": embedding}
                for phrase, embedding in zip(phrases_to_embed, new_embeddings)
            ]
            lance_db.add_embeddings(table_name, new_data)
            for item in new_data:
                existing_embeddings[item["text_hash"]] = np.array(item["vector"])

    return existing_embeddings, total_processing_time


def run_test(
    rows: List[Tuple[str, str, str, str]],
    db: EmbeddingDatabase,
    lance_db: LanceDBClient,
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
    Processes a test run using SQLite for metadata and LanceDB for embeddings.
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
        client = ProductionEmbeddingClient(
            model_config["base_url"],
            model_name,
            expected_dimension=model_config.get("dimensions"),
            timeout=model_config.get("timeout", 30),
        )
        embedding_function = client.get_embeddings

    embed_themes_list, _ = embedding_function(themes)
    embed_themes = np.array(embed_themes_list)

    unprocessed_rows = [row for row in rows if row[0] not in processed_files]

    for item in tqdm(unprocessed_rows, desc=f"Processing {model_name}", leave=False, disable=not show_progress):
        identifiant, _, _, texte = item
        if not texte or not texte.strip():
            continue

        item_phrases = chunk_text(texte, chunk_size, chunk_overlap, chunking_strategy)
        if not item_phrases:
            continue

        model_dimensions = model_config.get("dimensions")
        if model_dimensions is None:
            raise ValueError(f"Model dimensions not configured for {model_name}")

        all_embeddings_map, p_time = _get_or_create_embeddings(
            lance_db,
            embedding_function,
            item_phrases,
            model_name,
            model_dimensions,
        )

        phrase_hashes = {p: get_text_hash(p) for p in item_phrases}
        item_embed_phrases = np.array(
            [
                all_embeddings_map[h]
                for p in item_phrases
                if (h := phrase_hashes[p]) in all_embeddings_map
            ]
        )

        if item_embed_phrases.size == 0:
            continue

        similarites = calculate_similarity(embed_themes, item_embed_phrases, similarity_metric)
        labels = np.argmax(similarites, axis=0)

        cohesion_sep = calculate_cohesion_separation(item_embed_phrases, labels)
        clustering_metrics = calculate_clustering_metrics(item_embed_phrases, labels)

        results["files"][identifiant] = {
            "phrases": item_phrases,
            "themes": themes,
            "similarities": similarites.max(axis=0),
            "metrics": {
                **cohesion_sep,
                **clustering_metrics,
                "processing_time": p_time,
                "mean_similarity": float(np.mean(similarites.max(axis=0))),
            },
        }

    return {"results": results}
