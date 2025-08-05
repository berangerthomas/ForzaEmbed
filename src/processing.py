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

from .database import EmbeddingDatabase
from .embedding_client import ProductionEmbeddingClient
from .evaluation_metrics import calculate_all_metrics
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
        pbar: Any,  # Progress bar object
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

        unprocessed_rows = [row for row in rows if row[0] not in processed_files]

        if not themes_embeddings_map:
            logging.error(
                f"Failed to embed themes for model '{model_name}'. "
                f"Skipping {len(unprocessed_rows)} files for this combination."
            )
            pbar.update(len(unprocessed_rows))
            return {"results": {}}

        theme_hashes = [self.get_text_hash(theme) for theme in themes]
        embed_themes_list = [
            themes_embeddings_map[h] for h in theme_hashes if h in themes_embeddings_map
        ]
        embed_themes = np.array(embed_themes_list)

        for item in unprocessed_rows:
            identifiant, name, location_type, texte = item
            file_info = f"{identifiant} {location_type[:5]} {name[-30:]}"
            grid_params = (
                f"cs{chunk_size} co{chunk_overlap} {similarity_metric[:3]} "
                f"{chunking_strategy} {theme_name[-13:]}"
            )
            description = f"{file_info} | {model_name[-30:]} | {grid_params}"
            pbar.set_description(description[:100])
            if not texte or not texte.strip():
                pbar.update(1)
                continue

            item_phrases = chunk_text(
                texte, chunk_size, chunk_overlap, chunking_strategy
            )
            if not item_phrases:
                pbar.update(1)
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
                pbar.update(1)
                continue

            if embed_themes.shape[1] != item_embed_phrases.shape[1]:
                logging.error(f"Dimension mismatch for file '{identifiant}'.")
                pbar.update(1)
                continue

            similarites = self.calculate_similarity(
                embed_themes, item_embed_phrases, similarity_metric
            )
            labels = np.argmax(similarites, axis=0)

            all_metrics = calculate_all_metrics(
                embed_themes, item_embed_phrases, labels
            )

            results["files"][identifiant] = {
                "phrases": item_phrases,
                "similarities": similarites.max(axis=0),
                "metrics": {
                    **all_metrics,
                    "processing_time": p_time,
                    "mean_similarity": float(np.mean(similarites.max(axis=0))),
                },
            }
            pbar.update(1)

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

    def refresh_all_metrics(self):
        """
        Recalculates and updates evaluation metrics for all runs in the database.
        """
        all_run_names = self.db.get_all_run_names()
        logging.info(f"Found {len(all_run_names)} runs to refresh.")

        for run_name in all_run_names:
            logging.info(f"Refreshing metrics for run: {run_name}")
            run_details = self.db.get_run_details(run_name)
            if not run_details:
                logging.warning(f"Could not find details for run: {run_name}")
                continue

            # Get themes for the run
            theme_name = run_details["theme_name"]
            themes = self.config["grid_search_params"]["themes"].get(theme_name)
            if not themes:
                logging.warning(f"Themes '{theme_name}' not found in config.")
                continue

            # Get theme embeddings
            embedding_function = self._get_embedding_function(
                {"type": run_details["model_type"], "name": run_details["model_name"]}
            )
            themes_embeddings_map, _ = self._get_or_create_embeddings(
                embedding_function, run_details["model_name"], themes
            )
            theme_hashes = [self.get_text_hash(theme) for theme in themes]
            embed_themes_list = [
                themes_embeddings_map[h]
                for h in theme_hashes
                if h in themes_embeddings_map
            ]
            embed_themes = np.array(embed_themes_list)

            # Get all processed files for the run
            processed_files_data = self.db.get_all_processing_results_for_run(run_name)

            for file_id, file_data in processed_files_data.items():
                item_phrases = file_data["phrases"]
                if not item_phrases:
                    continue

                # Get phrase embeddings
                all_embeddings_map, _ = self._get_or_create_embeddings(
                    embedding_function, run_details["model_name"], item_phrases
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

                # Recalculate similarities and metrics
                similarites = self.calculate_similarity(
                    embed_themes,
                    item_embed_phrases,
                    run_details["similarity_metric"],
                )
                labels = np.argmax(similarites, axis=0)
                all_metrics = calculate_all_metrics(
                    embed_themes, item_embed_phrases, labels
                )

                # Update the database
                self.db.update_metrics_for_file(run_name, file_id, all_metrics)
        logging.info("Finished refreshing all metrics.")
