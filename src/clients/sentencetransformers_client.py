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

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ..utils.embedding_pooling import pool_embeddings, split_text_into_chunks


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
        cls, 
        texts: list[str], 
        model_name: str, 
        expected_dimension: int | None = None,
        batch_size: int = 32,
        max_tokens: int | None = None,
        pooling_strategy: str = "max"
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts using a local model.

        Automatically adds prefix for Jina models.

        Args:
            texts: List of texts to embed.
            model_name: Name of the sentence-transformer model.
            expected_dimension: Expected embedding dimension for validation.
            batch_size: Number of texts to process at once (lower = less memory).
            max_tokens: Maximum number of tokens per text. When a text exceeds this 
                limit, it will be split into chunks and recombined using the 
                pooling_strategy. If None, uses model default.
            pooling_strategy: Strategy for combining chunk embeddings when text 
                exceeds max_tokens. Options: "max" (default), "average", "weighted".

        Returns:
            List of embedding vectors as lists of floats.

        Raises:
            ValueError: If embedding dimension doesn't match expected.
        """
        instance = cls.get_instance(model_name)
        if "jina" in model_name:
            texts = ["search_document: " + text for text in texts]
        
        # If no max_tokens specified, use default behavior (truncation)
        if max_tokens is None:
            embeddings = instance.encode(texts, convert_to_tensor=False, batch_size=batch_size)
            embeddings = embeddings.tolist()
        else:
            # Dynamic chunking for long texts
            embeddings = []
            chunked_count = 0
            
            for text in texts:
                text_token_count = len(text.split())
                
                if text_token_count <= max_tokens:
                    # Text fits, embed directly
                    emb = instance.encode([text], convert_to_tensor=False, batch_size=1)
                    embeddings.append(emb[0].tolist())
                else:
                    # Text too long, need to chunk and pool
                    chunked_count += 1
                    text_chunks = split_text_into_chunks(text, max_tokens)
                    chunk_embeddings = instance.encode(text_chunks, convert_to_tensor=False, batch_size=batch_size)
                    chunk_embeddings = [np.array(e) for e in chunk_embeddings]
                    pooled = pool_embeddings(chunk_embeddings, strategy=pooling_strategy)
                    embeddings.append(pooled.tolist())
            
            if chunked_count > 0:
                tqdm.write(
                    f"⚠️ Dynamic chunking: {chunked_count} text(s) split and pooled "
                    f"({pooling_strategy} pooling) due to token limit ({max_tokens})"
                )

        if expected_dimension and embeddings:
            actual_dimension = len(embeddings[0])
            if actual_dimension != expected_dimension:
                raise ValueError(
                    f"Expected dimension {expected_dimension}, but got {actual_dimension} for model {model_name}"
                )

        return embeddings
