import os
import time
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
        elif "voyage" in model.lower():
            api_key = os.environ.get("API_KEY_VOYAGEAI")
        else:
            api_key = os.environ.get("API_KEY_OPENAI")

        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    # Récupère les embeddings pour une liste de textes via l'API.
    def get_embeddings(self, texts: List[str]) -> tuple[List[List[float]], float]:
        """
        Récupère les embeddings pour une liste de textes via l'API.

        Args:
            texts (List[str]): Liste de textes.

        Returns:
            tuple[List[List[float]], float]: Liste des vecteurs d'embedding et temps de réponse.
        """
        if not texts:
            return [], 0.0
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model, "input": texts}
        start_time = time.time()
        try:
            response = self.session.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            embeddings = [data["embedding"] for data in result["data"]]
            end_time = time.time()
            return embeddings, end_time - start_time
        except requests.exceptions.RequestException as e:
            print(f"❌ API Error: {e}")
            return [], 0.0
        except (KeyError, IndexError) as e:
            print(f"❌ API Response Parsing Error: {e}")
            return [], 0.0
