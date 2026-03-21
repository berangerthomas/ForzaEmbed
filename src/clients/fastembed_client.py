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

import numpy as np
import onnxruntime as ort
from fastembed import TextEmbedding
from tqdm import tqdm

from ..utils.embedding_pooling import pool_embeddings, split_text_into_chunks


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
        GPU detection is re-attempted each time to allow for dynamic GPU availability.

        Args:
            model_name: Name of the FastEmbed model.

        Returns:
            Loaded TextEmbedding model instance.
        """
        if model_name not in cls._instances:
            cpu_count = os.cpu_count()
            
            # Check if CUDA provider is available in onnxruntime (re-check each time)
            available_providers = ort.get_available_providers()
            cuda_available = "CUDAExecutionProvider" in available_providers
            
            if cuda_available:
                try:
                    tqdm.write(f"🚀 Attempting to load FastEmbed model: {model_name} with GPU support")
                    cls._instances[model_name] = TextEmbedding(
                        model_name, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                    )
                    tqdm.write("✅ GPU detected and configured successfully.")
                except Exception as e:
                    tqdm.write(f"⚠️ GPU initialization failed ({e}), falling back to CPU.")
                    tqdm.write(
                        f"🚀 Loading FastEmbed model: {model_name} with {cpu_count} CPU threads"
                    )
                    cls._instances[model_name] = TextEmbedding(
                        model_name, providers=["CPUExecutionProvider"], threads=cpu_count
                    )
            else:
                tqdm.write("⚠️ CUDAExecutionProvider not available in onnxruntime, using CPU.")
                tqdm.write(
                    f"🚀 Loading FastEmbed model: {model_name} with {cpu_count} CPU threads"
                )
                cls._instances[model_name] = TextEmbedding(
                    model_name, providers=["CPUExecutionProvider"], threads=cpu_count
                )
        return cls._instances[model_name]
    
    @classmethod
    def reset_instance(cls, model_name: str) -> None:
        """Reset a model instance to allow reloading with different settings.
        
        Useful when GPU becomes available/unavailable and we want to re-attempt
        GPU loading.
        
        Args:
            model_name: Name of the FastEmbed model to reset.
        """
        if model_name in cls._instances:
            del cls._instances[model_name]
            tqdm.write(f"🔄 Reset FastEmbed model instance: {model_name}")

    @staticmethod
    def get_embeddings(
        texts: list[str], 
        model_name: str, 
        expected_dimension: int | None = None,
        batch_size: int = 32,
        max_tokens: int | None = None,
        pooling_strategy: str = "max"
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed.
            model_name: Name of the FastEmbed model to use.
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
        instance = FastEmbedClient.get_instance(model_name)
        
        # If no max_tokens specified, use default behavior
        if max_tokens is None:
            embeddings = list(instance.embed(texts, batch_size=batch_size))
            embeddings = [np.array(e) for e in embeddings]
        else:
            # Dynamic chunking for long texts
            embeddings = []
            chunked_count = 0
            
            for text in texts:
                text_token_count = len(text.split())
                
                if text_token_count <= max_tokens:
                    # Text fits, embed directly
                    emb = list(instance.embed([text], batch_size=1))
                    embeddings.append(np.array(emb[0]))
                else:
                    # Text too long, need to chunk and pool
                    chunked_count += 1
                    text_chunks = split_text_into_chunks(text, max_tokens)
                    chunk_embeddings = list(instance.embed(text_chunks, batch_size=batch_size))
                    chunk_embeddings = [np.array(e) for e in chunk_embeddings]
                    pooled = pool_embeddings(chunk_embeddings, strategy=pooling_strategy)
                    embeddings.append(pooled)
            
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

        return [e.tolist() for e in embeddings]
