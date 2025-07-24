import os
from typing import List

import requests
from dotenv import load_dotenv

load_dotenv()


# Classe client pour obtenir des embeddings via une API de production.
class ProductionEmbeddingClient:
    """
    Client pour obtenir des embeddings depuis une API de production.

    Args:
        base_url (str): URL de base de l'API.
        model (str): Nom du modèle d'embedding à utiliser.
    """

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        # Détermine la clé API à utiliser en fonction du nom du modèle
        if "mistral" in model.lower():
            api_key = os.environ.get("API_KEY_MISTRAL")
        else:
            api_key = os.environ.get("API_KEY_OPENAI")

        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    # Récupère les embeddings pour une liste de textes via l'API.
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Récupère les embeddings pour une liste de textes via l'API.

        Args:
            texts (List[str]): Liste de textes.

        Returns:
            List[List[float]]: Liste des vecteurs d'embedding.
        """
        if not texts:
            return []
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model, "input": texts}
        try:
            response = self.session.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return [data["embedding"] for data in result["data"]]
        except requests.exceptions.RequestException as e:
            print(f"❌ API Error: {e}")
            return []
        except (KeyError, IndexError) as e:
            print(f"❌ API Response Parsing Error: {e}")
            return []
