"""Transformers client for local embedding generation.

This module provides a client for generating embeddings using the Hugging Face
transformers library directly, with special handling for Jina models.

Example:
    Generate embeddings using Transformers::

        from src.clients.transformers_client import TransformersClient

        embeddings = TransformersClient.get_embeddings(
            texts=["Hello world"],
            model_name="BAAI/bge-small-en-v1.5"
        )
"""

from typing import Dict, Tuple

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def mean_pooling(
    token_embeddings: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Perform mean pooling on token embeddings.

    Args:
        token_embeddings: Tensor of token-level embeddings.
        attention_mask: Attention mask for the input tokens.

    Returns:
        Mean-pooled sentence embeddings tensor.
    """
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class TransformersClient:
    """Client for managing local transformers embedding models.

    Implements singleton pattern for model instances with special handling
    for Jina models and their task labels.

    Attributes:
        _instances: Class-level cache of loaded model and tokenizer instances.
    """

    _instances: Dict[str, Tuple[PreTrainedModel, PreTrainedTokenizer]] = {}

    @classmethod
    def get_instance(
        cls, model_name: str
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Get or create a transformers model and tokenizer instance.

        Args:
            model_name: Name of the transformers model.

        Returns:
            Tuple of (model, tokenizer) instances.
        """
        if model_name not in cls._instances:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            cls._instances[model_name] = (model, tokenizer)
        return cls._instances[model_name]

    @classmethod
    def get_embeddings(
        cls, texts: list[str], model_name: str, expected_dimension: int | None = None
    ) -> list[list[float]]:
        """Generate embeddings using a local transformers model.

        Handles special cases for Jina models including task labels and
        different output formats.

        Args:
            texts: List of texts to embed.
            model_name: Name of the transformers model.
            expected_dimension: Expected embedding dimension for validation.

        Returns:
            List of normalized embedding vectors as lists of floats.

        Raises:
            ValueError: If embedding dimension doesn't match expected or
                embeddings cannot be extracted.
        """
        model, tokenizer = cls.get_instance(model_name)

        # Tokenize sentences
        encoded_input = tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )

        # Compute token embeddings
        with torch.no_grad():
            if "jina" in model_name.lower():
                # Les modèles Jina nécessitent obligatoirement task_label
                try:
                    model_output = model(**encoded_input, task_label="text-matching")
                except Exception:
                    # Fallback sans task_label pour les anciens modèles Jina
                    model_output = model(**encoded_input)

                # Pour Jina v4, utiliser single_vec_emb qui contient déjà les embeddings poolés
                if (
                    hasattr(model_output, "single_vec_emb")
                    and model_output.single_vec_emb is not None
                ):
                    sentence_embeddings = model_output.single_vec_emb
                    sentence_embeddings = torch.nn.functional.normalize(
                        sentence_embeddings, p=2, dim=1
                    )
                    embeddings = sentence_embeddings.tolist()

                    if expected_dimension and embeddings:
                        actual_dimension = len(embeddings[0])
                        if actual_dimension != expected_dimension:
                            raise ValueError(
                                f"Expected dimension {expected_dimension}, but got {actual_dimension} for model {model_name}"
                            )
                    return embeddings

                # Fallback pour les anciens modèles Jina ou si single_vec_emb n'est pas disponible
                token_embeddings = None
                if (
                    hasattr(model_output, "last_hidden_state")
                    and model_output.last_hidden_state is not None
                ):
                    token_embeddings = model_output.last_hidden_state
                elif (
                    hasattr(model_output, "vlm_last_hidden_states")
                    and model_output.vlm_last_hidden_states is not None
                ):
                    token_embeddings = model_output.vlm_last_hidden_states
                elif (
                    hasattr(model_output, "pooler_output")
                    and model_output.pooler_output is not None
                ):
                    sentence_embeddings = model_output.pooler_output
                    sentence_embeddings = torch.nn.functional.normalize(
                        sentence_embeddings, p=2, dim=1
                    )
                    embeddings = sentence_embeddings.tolist()

                    if expected_dimension and embeddings:
                        actual_dimension = len(embeddings[0])
                        if actual_dimension != expected_dimension:
                            raise ValueError(
                                f"Expected dimension {expected_dimension}, but got {actual_dimension} for model {model_name}"
                            )
                    return embeddings

                if token_embeddings is None:
                    raise ValueError(
                        f"Unable to extract embeddings from Jina model output for {model_name}. Available attributes: {[attr for attr in dir(model_output) if not attr.startswith('_')]}"
                    )
