"""FastEmbed client for local embedding generation.

This module provides a client for generating embeddings using the FastEmbed
library with GPU acceleration support and automatic fallback to CPU.

Example:
    Generate embeddings using FastEmbed::

        from src.clients.fastembed_client import FastEmbedClient

        embeddings = FastEmbedClient.get_embeddings(
            texts=["Hello world"],
            model_name="BAAI/bge-small-en-v1.5"
        )
"""

import os

from fastembed import TextEmbedding
from tqdm import tqdm


class FastEmbedClient:
    """Client for managing FastEmbed embedding models.

    Implements singleton pattern for model instances to avoid reloading.
    Supports GPU acceleration with automatic CPU fallback.

    Attributes:
        _instances: Class-level cache of loaded model instances.
    """

    _instances: dict[str, TextEmbedding] = {}

    @classmethod
    def get_instance(cls, model_name: str) -> TextEmbedding:
        """Get or create a FastEmbed model instance.

        Attempts GPU acceleration first, falls back to CPU if unavailable.

        Args:
            model_name: Name of the FastEmbed model.

        Returns:
            Loaded TextEmbedding model instance.
        """
        if model_name not in cls._instances:
            try:
                # Try to use GPU first
                tqdm.write(f"🚀 Attempting to load FastEmbed model: {model_name} with GPU support")
                cls._instances[model_name] = TextEmbedding(
                    model_name, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
                tqdm.write("✅ GPU detected and configured successfully.")
            except Exception as e:
                tqdm.write(f"⚠️ GPU not available ({e}), falling back to CPU.")
                # Fallback to CPU with multi-threading
                cpu_count = os.cpu_count()
                tqdm.write(
                    f"🚀 Loading FastEmbed model: {model_name} with {cpu_count} CPU threads"
                )
                cls._instances[model_name] = TextEmbedding(
                    model_name, providers=["CPUExecutionProvider"], threads=cpu_count
                )
        return cls._instances[model_name]

    @staticmethod
    def get_embeddings(
        texts: list[str], model_name: str, expected_dimension: int | None = None
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed.
            model_name: Name of the FastEmbed model to use.
            expected_dimension: Expected embedding dimension for validation.

        Returns:
            List of embedding vectors as lists of floats.

        Raises:
            ValueError: If embedding dimension doesn't match expected.
        """
        instance = FastEmbedClient.get_instance(model_name)
        embeddings = list(instance.embed(texts))

        if expected_dimension and embeddings:
            actual_dimension = len(embeddings[0])
            if actual_dimension != expected_dimension:
                raise ValueError(
                    f"Expected dimension {expected_dimension}, but got {actual_dimension} for model {model_name}"
                )

        return embeddings
