"""Hugging Face embedding client using transformers library.

This module provides functions for generating embeddings using generic
Hugging Face models with mean pooling and normalization.

Example:
    Generate embeddings using a Hugging Face model::

        from src.clients.huggingface_client import get_huggingface_embeddings

        embeddings = get_huggingface_embeddings(
            texts=["Hello world"],
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
"""

from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def mean_pooling(
    model_output: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Perform mean pooling on token embeddings to get sentence embedding.

    Args:
        model_output: Model output containing token embeddings.
        attention_mask: Attention mask for the input tokens.

    Returns:
        Mean-pooled sentence embeddings tensor.
    """
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def get_huggingface_embeddings(
    texts: List[str], model_name: str, expected_dimension: int | None = None
) -> List[List[float]]:
    """Generate embeddings using a generic Hugging Face model.

    Loads the model and tokenizer, processes texts, and applies mean pooling
    with L2 normalization.

    Args:
        texts: List of texts to embed.
        model_name: Name of the Hugging Face model to use.
        expected_dimension: Expected embedding dimension for validation.

    Returns:
        List of normalized embedding vectors as lists of floats.

    Raises:
        ValueError: If embedding dimension doesn't match expected.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # For instruction-tuned models, add a prefix
        if "instruct" in model_name:
            texts = [f"passage: {text}" for text in texts]

        encoded_input = tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            model_output = model(**encoded_input)

        sentence_embeddings = mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        # Normalize embeddings
        normalized_embeddings = torch.nn.functional.normalize(
            sentence_embeddings, p=2, dim=1
        )

        if expected_dimension and normalized_embeddings.shape[0] > 0:
            actual_dimension = normalized_embeddings.shape[1]
            if actual_dimension != expected_dimension:
                raise ValueError(
                    f"Expected dimension {expected_dimension}, but got {actual_dimension} for model {model_name}"
                )

        return normalized_embeddings.tolist()

    except Exception as e:
        tqdm.write(f"❌ Error getting Hugging Face embeddings for {model_name}: {e}")
        return []
