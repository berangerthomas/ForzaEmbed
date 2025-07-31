import os

from fastembed import TextEmbedding
from tqdm import tqdm


class FastEmbedClient:
    """
    Client to manage FastEmbed embedding models.
    """

    _instances = {}

    @classmethod
    def get_instance(cls, model_name: str):
        if model_name not in cls._instances:
            cpu_count = os.cpu_count()
            tqdm.write(
                f"🚀 Loading FastEmbed model: {model_name} with {cpu_count} threads"
            )
            cls._instances[model_name] = TextEmbedding(model_name, threads=cpu_count)
        return cls._instances[model_name]

    @staticmethod
    def get_embeddings(
        texts: list[str], model_name: str, expected_dimension: int | None = None
    ) -> tuple[list[list[float]], float]:
        import time

        instance = FastEmbedClient.get_instance(model_name)
        start_time = time.time()
        embeddings = list(instance.embed(texts))

        if expected_dimension and embeddings:
            actual_dimension = len(embeddings[0])
            if actual_dimension != expected_dimension:
                raise ValueError(
                    f"Expected dimension {expected_dimension}, but got {actual_dimension} for model {model_name}"
                )

        end_time = time.time()
        return embeddings, end_time - start_time
