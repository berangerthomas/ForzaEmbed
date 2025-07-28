from typing import Dict

from sentence_transformers import SentenceTransformer


class LocalEmbeddingClient:
    """
    Client pour gérer les modèles d'embedding locaux en tant que singletons.
    """

    _instances: Dict[str, SentenceTransformer] = {}

    @classmethod
    def get_instance(cls, model_name: str) -> SentenceTransformer:
        """
        Récupère une instance du modèle d'embedding.
        Si l'instance n'existe pas, elle est créée.
        """
        if model_name not in cls._instances:
            print(f"🚀 Chargement du modèle local : {model_name}")
            cls._instances[model_name] = SentenceTransformer(model_name)
        return cls._instances[model_name]

    @classmethod
    def get_embeddings(
        cls, texts: list[str], model_name: str
    ) -> tuple[list[list[float]], float]:
        """
        Génère les embeddings pour une liste de textes en utilisant un modèle local.
        """
        import time

        instance = cls.get_instance(model_name)
        start_time = time.time()
        embeddings = instance.encode(texts, convert_to_tensor=False).tolist()
        end_time = time.time()

        return embeddings, end_time - start_time
