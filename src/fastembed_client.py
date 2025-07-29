from fastembed import TextEmbedding


class FastEmbedClient:
    """
    Client pour gérer les modèles d'embedding FastEmbed.
    """

    _instances = {}

    @classmethod
    def get_instance(cls, model_name: str):
        if model_name not in cls._instances:
            # print(f"🚀 Chargement du modèle FastEmbed : {model_name}")
            cls._instances[model_name] = TextEmbedding(model_name)
        return cls._instances[model_name]

    @classmethod
    def get_embeddings(
        cls, texts: list[str], model_name: str
    ) -> tuple[list[list[float]], float]:
        import time

        instance = cls.get_instance(model_name)
        start_time = time.time()
        embeddings = list(instance.embed(texts))
        end_time = time.time()
        return embeddings, end_time - start_time
