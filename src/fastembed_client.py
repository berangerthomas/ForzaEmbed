from fastembed import TextEmbedding


class FastEmbedClient:
    """
    Client to manage FastEmbed embedding models.
    """

    _instances = {}

    @classmethod
    def get_instance(cls, model_name: str):
        if model_name not in cls._instances:
            # print(f"🚀 Loading FastEmbed model: {model_name}")
            cls._instances[model_name] = TextEmbedding(model_name)
        return cls._instances[model_name]

    @classmethod
    def get_embeddings(
        cls, texts: list[str], model_name: str, expected_dimension: int | None = None
    ) -> tuple[list[list[float]], float]:
        import time

        instance = cls.get_instance(model_name)
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
