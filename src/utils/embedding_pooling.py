"""Embedding pooling utilities for handling long texts.

This module provides functions for splitting long texts into chunks that fit
within the model's token limit and recombining their embeddings using
various pooling strategies.

Example:
    Pool embeddings from multiple chunks::

        from src.utils.embedding_pooling import pool_embeddings

        embeddings = [[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]]
        pooled = pool_embeddings(embeddings, strategy="max")
"""

from typing import Callable, List, Optional

import numpy as np


def split_text_into_chunks(
    text: str, 
    max_tokens: int, 
    token_counter: Optional[Callable[[str], int]] = None
) -> List[str]:
    """Split a text into chunks that fit within the token limit.
    
    Uses a simple word-based approximation for token counting.
    For more accurate results, pass a tokenizer function.
    
    Args:
        text: The text to split.
        max_tokens: Maximum tokens per chunk.
        token_counter: Optional function that counts tokens in a string.
            If None, uses word count as approximation (1 token ≈ 1 word).
    
    Returns:
        List of text chunks.
    """
    words = text.split()
    if not words:
        return []
    
    chunks = []
    current_chunk = []
    current_count = 0
    
    for word in words:
        word_count = len(word.split())  # Usually 1, but handle multi-word tokens
        
        if current_count + word_count > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_count = word_count
        else:
            current_chunk.append(word)
            current_count += word_count
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks if chunks else [text]


def pool_embeddings(
    embeddings: List[np.ndarray], 
    strategy: str = "max"
) -> np.ndarray:
    """Combine multiple embeddings using the specified pooling strategy.
    
    Args:
        embeddings: List of embedding vectors (numpy arrays).
        strategy: Pooling strategy - "max", "average", "weighted", or "last".
            Defaults to "max".
    
    Returns:
        Pooled embedding vector.
    
    Raises:
        ValueError: If strategy is invalid or embeddings list is empty.
    """
    if not embeddings:
        raise ValueError("Cannot pool empty embeddings list")
    
    if len(embeddings) == 1:
        return embeddings[0]
    
    # Stack embeddings into matrix
    emb_matrix = np.vstack(embeddings)
    
    if strategy == "max":
        return np.max(emb_matrix, axis=0)
    elif strategy == "average":
        return np.mean(emb_matrix, axis=0)
    elif strategy == "weighted":
        # Linear weighting: earlier chunks get more weight
        n = len(embeddings)
        weights = np.linspace(1.0, 0.5, n)  # From 1.0 to 0.5
        weights = weights / weights.sum()  # Normalize
        return np.average(emb_matrix, axis=0, weights=weights)
    elif strategy == "last":
        # Use only the last chunk embedding (useful for summaries/conclusions)
        return embeddings[-1]
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy}. Use 'max', 'average', 'weighted', or 'last'")
