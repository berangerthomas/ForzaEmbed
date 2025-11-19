
import hashlib
import logging
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from ..clients.api_client import ProductionEmbeddingClient
from ..clients.fastembed_client import FastEmbedClient
from ..clients.huggingface_client import get_huggingface_embeddings
from ..clients.sentencetransformers_client import SentenceTransformersClient
from ..clients.transformers_client import TransformersClient
from ..utils.database import EmbeddingDatabase
from ..core.config import AppConfig

class EmbeddingService:
    """
    Handles embedding generation and caching.
    """
    def __init__(self, db: EmbeddingDatabase, config: AppConfig):
        self.db = db
        self.config = config
        self.multiprocessing_config = self.config.multiprocessing

    def get_embedding_function(self, model_config: Any) -> Callable:
        """Creates the appropriate embedding function based on model type."""
        model_type = model_config.type
        model_name = model_config.name

        # Map model types to their respective embedding functions/clients
        local_model_map = {
            "fastembed": FastEmbedClient.get_embeddings,
            "huggingface": get_huggingface_embeddings,
            "sentence_transformers": SentenceTransformersClient.get_embeddings,
            "transformers": TransformersClient.get_embeddings,
        }

        if model_type in local_model_map:
            embedding_func = local_model_map[model_type]

            def get_embeddings(texts):
                return embedding_func(
                    texts,
                    model_name=model_name,
                    expected_dimension=model_config.dimensions,
                )

            return get_embeddings

        # Handle API models
        if model_type == "api":
            api_batch_sizes = self.multiprocessing_config.api_batch_sizes
            model_lower = model_name.lower()
            batch_size = api_batch_sizes.get("default", 100)
            for provider, size in api_batch_sizes.items():
                if provider in model_lower:
                    batch_size = size
                    break

            if not model_config.base_url:
                raise ValueError(f"API model '{model_name}' requires a base_url.")

            client = ProductionEmbeddingClient(
                model_config.base_url,
                model_name,
                expected_dimension=model_config.dimensions,
                timeout=model_config.timeout or 30,
                initial_batch_size=batch_size,
            )
            return client.get_embeddings

        raise ValueError(f"Unsupported model type: {model_type}")

    def get_or_create_embeddings(
        self,
        embedding_function: Callable,
        base_model_name: str,
        phrases: List[str],
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Retrieves embeddings from cache or generates and caches them.
        
        Returns:
            tuple: (embeddings_dict, computation_time_seconds)
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

        computation_time = 0.0
        if phrases_to_embed:
            start_time = time.perf_counter()
            new_embeddings_list = embedding_function(phrases_to_embed)
            computation_time = time.perf_counter() - start_time

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
        return all_embeddings_for_phrases, computation_time

    @staticmethod
    def get_text_hash(text: str) -> str:
        """Generates a SHA-256 hash for a given text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
