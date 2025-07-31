import os
import time
from typing import List

import requests
from dotenv import load_dotenv

load_dotenv()


# Client class to obtain embeddings via a production API.
class ProductionEmbeddingClient:
    """
    Client to obtain embeddings from a production API.

    Args:
        base_url (str): Base URL of the API.
        model (str): Name of the embedding model to use.
        expected_dimension (int, optional): The expected dimension of the embeddings.
        timeout (int, optional): Timeout for the request in seconds. Defaults to 30.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        expected_dimension: int | None = None,
        timeout: int = 30,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.expected_dimension = expected_dimension
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        # Determines which API key to use based on the model name
        if "mistral" in model.lower():
            api_key = os.environ.get("API_KEY_MISTRAL")
        elif "voyage" in model.lower():
            api_key = os.environ.get("API_KEY_VOYAGEAI")
        else:
            api_key = os.environ.get("API_KEY_OPENAI")

        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    # Retrieves embeddings for a list of texts via the API.
    def get_embeddings(self, texts: List[str]) -> tuple[List[List[float]], float]:
        """
        Retrieves embeddings for a list of texts via the API.

        Args:
            texts (List[str]): List of texts.

        Returns:
            tuple[List[List[float]], float]: List of embedding vectors and response time.
        """
        if not texts:
            return [], 0.0
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model, "input": texts}
        start_time = time.time()
        try:
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            embeddings = [data["embedding"] for data in result["data"]]

            if self.expected_dimension and embeddings:
                actual_dimension = len(embeddings[0])
                if actual_dimension != self.expected_dimension:
                    raise ValueError(
                        f"Expected dimension {self.expected_dimension}, but got {actual_dimension} for model {self.model}"
                    )

            end_time = time.time()
            return embeddings, end_time - start_time
        except requests.exceptions.RequestException as e:
            print(f"❌ API Error: {e}")
            return [], 0.0
        except (KeyError, IndexError) as e:
            print(f"❌ API Response Parsing Error: {e}")
            return [], 0.0
