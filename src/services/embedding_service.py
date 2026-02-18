"""Embedding service for ForzaEmbed.

This module provides the EmbeddingService class that handles embedding
generation and caching. It abstracts the different embedding clients and
provides a unified interface for the processing pipeline.

Example:
    Generate embeddings using the service::

        from src.services.embedding_service import EmbeddingService

        service = EmbeddingService(db, config)
        embed_func = service.get_embedding_function(model_config)
        embeddings, time = service.get_or_create_embeddings(embed_func, "model", texts)
"""

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
from ..core.config import AppConfig, ModelConfig
from ..utils.database import EmbeddingDatabase

# Type alias for embedding functions
EmbeddingFunc = Callable[[List[str]], List[List[float]]]


class EmbeddingService:
    """Handle embedding generation and caching.

    Provides a unified interface for generating embeddings using different
    backends (API, FastEmbed, Sentence Transformers, etc.) with automatic
    caching.

    Attributes:
        db: The embedding database for caching.
        config: The application configuration.
        multiprocessing_config: Multiprocessing settings from config.
    """

    def __init__(self, db: EmbeddingDatabase, config: AppConfig) -> None:
        """Initialize the EmbeddingService.

        Args:
            db: The embedding database for caching.
            config: The application configuration.
        """
        self.db = db
        self.config = config
        self.multiprocessing_config = self.config.multiprocessing

    def get_embedding_function(self, model_config: ModelConfig) -> EmbeddingFunc:
        """Create the appropriate embedding function based on model type.

        Args:
            model_config: Configuration for the embedding model.

        Returns:
            A callable that takes a list of texts and returns embeddings.

        Raises:
            ValueError: If the model type is unsupported or API model lacks base_url.
        """
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
        embedding_function: EmbeddingFunc,
        base_model_name: str,
        phrases: list[str],
    ) -> tuple[dict[str, np.ndarray], float]:
        """Retrieve embeddings from cache or generate and cache them.

        Checks the database cache for existing embeddings. For phrases not
        in cache, generates new embeddings using the provided function and
        stores them.

        Args:
            embedding_function: Function to generate embeddings for texts.
            base_model_name: Name of the embedding model for cache key.
            phrases: List of text phrases to embed.

        Returns:
            A tuple containing:
                - Dictionary mapping text hashes to embedding arrays.
                - Computation time in seconds for new embeddings.
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
        """Generate a SHA-256 hash for a given text.

        Args:
            text: The text to hash.

        Returns:
            Hexadecimal string of the SHA-256 hash.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
