"""Sentence Transformers client for local embedding generation.

This module provides a client for generating embeddings using the
sentence-transformers library with singleton pattern for model caching.

Example:
    Generate embeddings using Sentence Transformers::

        from src.clients.sentencetransformers_client import SentenceTransformersClient

        embeddings = SentenceTransformersClient.get_embeddings(
            texts=["Hello world"],
            model_name="all-MiniLM-L6-v2"
        )
"""

from typing import Dict

from sentence_transformers import SentenceTransformer


class SentenceTransformersClient:
    """Client for managing local sentence-transformer models.

    Implements singleton pattern for model instances to avoid reloading.

    Attributes:
        _instances: Class-level cache of loaded model instances.
    """

    _instances: Dict[str, SentenceTransformer] = {}

    @classmethod
    def get_instance(cls, model_name: str) -> SentenceTransformer:
        """Get or create a SentenceTransformer model instance.

        Args:
            model_name: Name of the sentence-transformer model.

        Returns:
            Loaded SentenceTransformer model instance.
        """
        if model_name not in cls._instances:
            cls._instances[model_name] = SentenceTransformer(model_name)
        return cls._instances[model_name]

    @classmethod
    def get_embeddings(
        cls, texts: list[str], model_name: str, expected_dimension: int | None = None
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts using a local model.

        Automatically adds prefix for Jina models.

        Args:
            texts: List of texts to embed.
            model_name: Name of the sentence-transformer model.
            expected_dimension: Expected embedding dimension for validation.

        Returns:
            List of embedding vectors as lists of floats.

        Raises:
            ValueError: If embedding dimension doesn't match expected.
        """
        instance = cls.get_instance(model_name)
        if "jina" in model_name:
            texts = ["search_document: " + text for text in texts]
        embeddings = instance.encode(texts, convert_to_tensor=False).tolist()

        if expected_dimension and embeddings:
            actual_dimension = len(embeddings[0])
            if actual_dimension != expected_dimension:
                raise ValueError(
                    f"Expected dimension {expected_dimension}, but got {actual_dimension} for model {model_name}"
                )

        return embeddings
