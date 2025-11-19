from typing import Dict, Tuple

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def mean_pooling(token_embeddings, attention_mask):
    """
    Performs mean pooling on the token embeddings.
    """
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class TransformersClient:
    """
    Client to manage local embedding models from transformers library as singletons.
    """

    _instances: Dict[str, Tuple[PreTrainedModel, PreTrainedTokenizer]] = {}

    @classmethod
    def get_instance(
        cls, model_name: str
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Retrieves an instance of the embedding model and tokenizer.
        If the instance does not exist, it is created.
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
        """
        Generates embeddings for a list of texts using a local transformers model.
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
