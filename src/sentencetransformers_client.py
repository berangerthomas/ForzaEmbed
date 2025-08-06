from typing import Dict

from sentence_transformers import SentenceTransformer


class SentenceTransformersClient:
    """
    Client to manage local embedding models as singletons.
    """

    _instances: Dict[str, SentenceTransformer] = {}

    @classmethod
    def get_instance(cls, model_name: str) -> SentenceTransformer:
        """
        Retrieves an instance of the embedding model.
        If the instance does not exist, it is created.
        """
        if model_name not in cls._instances:
            # print(f"🚀 Loading local model: {model_name}")
            cls._instances[model_name] = SentenceTransformer(model_name)
        return cls._instances[model_name]

    @classmethod
    def get_embeddings(
        cls, texts: list[str], model_name: str, expected_dimension: int | None = None
    ) -> list[list[float]]:
        """
        Generates embeddings for a list of texts using a local model.
        """
        instance = cls.get_instance(model_name)
        embeddings = instance.encode(texts, convert_to_tensor=False).tolist()

        if expected_dimension and embeddings:
            actual_dimension = len(embeddings[0])
            if actual_dimension != expected_dimension:
                raise ValueError(
                    f"Expected dimension {expected_dimension}, but got {actual_dimension} for model {model_name}"
                )

        return embeddings
