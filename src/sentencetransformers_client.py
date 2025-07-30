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
        cls, texts: list[str], model_name: str
    ) -> tuple[list[list[float]], float]:
        """
        Generates embeddings for a list of texts using a local model.
        """
        import time

        instance = cls.get_instance(model_name)
        start_time = time.time()
        embeddings = instance.encode(texts, convert_to_tensor=False).tolist()
        end_time = time.time()

        return embeddings, end_time - start_time
