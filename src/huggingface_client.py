from typing import List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def mean_pooling(model_output, attention_mask):
    """
    Performs mean pooling on the last hidden state to get a sentence embedding.
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
    """
    Generates embeddings for a list of texts using a generic Hugging Face model.

    Args:
        texts (List[str]): The list of texts to embed.
        model_name (str): The name of the Hugging Face model to use.

    Returns:
        The list of embeddings.
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
