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
            try:
                # Try to use GPU first
                tqdm.write(f"🚀 Attempting to load FastEmbed model: {model_name} with GPU support")
                cls._instances[model_name] = TextEmbedding(
                    model_name, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
                tqdm.write("✅ GPU detected and configured successfully.")
            except Exception as e:
                tqdm.write(f"⚠️ GPU not available ({e}), falling back to CPU.")
                # Fallback to CPU with multi-threading
                cpu_count = os.cpu_count()
                tqdm.write(
                    f"🚀 Loading FastEmbed model: {model_name} with {cpu_count} CPU threads"
                )
                cls._instances[model_name] = TextEmbedding(
                    model_name, providers=["CPUExecutionProvider"], threads=cpu_count
                )
        return cls._instances[model_name]

    @staticmethod
    def get_embeddings(
        texts: list[str], model_name: str, expected_dimension: int | None = None
    ) -> list[list[float]]:
        instance = FastEmbedClient.get_instance(model_name)
        embeddings = list(instance.embed(texts))

        if expected_dimension and embeddings:
            actual_dimension = len(embeddings[0])
            if actual_dimension != expected_dimension:
                raise ValueError(
                    f"Expected dimension {expected_dimension}, but got {actual_dimension} for model {model_name}"
                )

        return embeddings
