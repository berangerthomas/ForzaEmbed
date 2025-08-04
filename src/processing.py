import hashlib
import logging
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
    pairwise_distances,
)
from tqdm import tqdm

from .database import EmbeddingDatabase
from .embedding_client import ProductionEmbeddingClient
from .evaluation_metrics import (
    calculate_clustering_metrics,
    calculate_cohesion_separation,
)
from .utils import chunk_text


class Processor:
    """
    Handles the core data processing logic for a single test run.
    """

    def __init__(self, db: EmbeddingDatabase, config: Dict[str, Any]):
        self.db = db
        self.config = config
        self.multiprocessing_config = config.get("multiprocessing", {})

    def run_test(
        self,
        rows: List[Tuple[str, str, str, str]],
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
        Processes a test run, handling embedding generation, similarity calculation,
        and metric evaluation.
        """
        model_type = model_config["type"]
        model_name = model_config["name"]
        results = {"files": {}}

        embedding_function = self._get_embedding_function(model_config)

        themes_embeddings_map, _ = self._get_or_create_embeddings(
            embedding_function, model_name, themes
        )
        if not themes_embeddings_map:
            logging.error(f"Failed to embed themes for model '{model_name}'.")
            return {}

        theme_hashes = [self.get_text_hash(theme) for theme in themes]
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

            item_phrases = chunk_text(
                texte, chunk_size, chunk_overlap, chunking_strategy
            )
            if not item_phrases:
                continue

            all_embeddings_map, p_time = self._get_or_create_embeddings(
                embedding_function, model_name, item_phrases
            )

            phrase_hashes = {p: self.get_text_hash(p) for p in item_phrases}
            item_embed_phrases = np.array(
                [
                    all_embeddings_map[h]
                    for p in item_phrases
                    if (h := phrase_hashes[p]) in all_embeddings_map
                ]
            )

            if item_embed_phrases.size == 0:
                continue

            if embed_themes.shape[1] != item_embed_phrases.shape[1]:
                logging.error(f"Dimension mismatch for file '{identifiant}'.")
                continue

            similarites = self.calculate_similarity(
                embed_themes, item_embed_phrases, similarity_metric
            )
            labels = np.argmax(similarites, axis=0)

            cohesion_sep = calculate_cohesion_separation(item_embed_phrases, labels)
            clustering_metrics = calculate_clustering_metrics(
                item_embed_phrases, labels
            )

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

    def _get_embedding_function(self, model_config: Dict[str, Any]) -> Callable:
        """Creates the appropriate embedding function based on model type."""
        model_type = model_config["type"]
        model_name = model_config["name"]

        if model_type in ["sentence_transformers", "fastembed", "huggingface"]:
            # This part will be further refactored to use a client factory
            def get_embeddings(texts):
                # Placeholder for future client factory
                from .fastembed_client import FastEmbedClient
                from .huggingface_client import get_huggingface_embeddings

                if model_type == "fastembed":
                    return FastEmbedClient.get_embeddings(
                        texts,
                        model_name=model_name,
                        expected_dimension=model_config.get("dimensions"),
                    )
                elif model_type == "huggingface":
                    return get_huggingface_embeddings(
                        texts,
                        model_name=model_name,
                        expected_dimension=model_config.get("dimensions"),
                    )
                return [], 0.0

            return get_embeddings
        else:  # API models
            api_batch_sizes = self.multiprocessing_config.get("api_batch_sizes", {})
            model_lower = model_name.lower()
            batch_size = api_batch_sizes.get("default", 100)
            for provider, size in api_batch_sizes.items():
                if provider in model_lower:
                    batch_size = size
                    break

            client = ProductionEmbeddingClient(
                model_config["base_url"],
                model_name,
                expected_dimension=model_config.get("dimensions"),
                timeout=model_config.get("timeout", 30),
            )
            client._initial_batch_size = batch_size
            return client.get_embeddings

    def _get_or_create_embeddings(
        self,
        embedding_function: Callable,
        base_model_name: str,
        phrases: List[str],
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Retrieves embeddings from cache or generates and caches them.
        """
        phrase_hashes = {phrase: self.get_text_hash(phrase) for phrase in phrases}
        existing_embeddings = self.db.get_embeddings_by_hashes(
            base_model_name, list(phrase_hashes.values())
        )
        phrases_to_embed = [
            phrase
            for phrase, h in phrase_hashes.items()
            if h not in existing_embeddings
        ]

        total_processing_time = 0.0
        if phrases_to_embed:
            new_embeddings_list, processing_time = embedding_function(phrases_to_embed)
            total_processing_time = processing_time

            if new_embeddings_list:
                new_embeddings_map = {
                    phrase_hashes[phrase]: np.array(embedding)
                    for phrase, embedding in zip(phrases_to_embed, new_embeddings_list)
                }
                self.db.save_embeddings_batch(base_model_name, new_embeddings_map)
                existing_embeddings.update(new_embeddings_map)

        all_embeddings_for_phrases = {
            h: existing_embeddings[h]
            for p, h in phrase_hashes.items()
            if h in existing_embeddings
        }
        return all_embeddings_for_phrases, total_processing_time

    @staticmethod
    def get_text_hash(text: str) -> str:
        """Generates a SHA-256 hash for a given text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
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
            distances = pairwise_distances(
                embed_themes, embed_phrases, metric="chebyshev"
            )
            return 1 / (1 + distances)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
