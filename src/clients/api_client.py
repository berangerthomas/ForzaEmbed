import json
import os
from typing import List

import requests
from dotenv import load_dotenv
from tqdm import tqdm

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
        initial_batch_size: int | None = None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.expected_dimension = expected_dimension
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._initial_batch_size = initial_batch_size

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
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Retrieves embeddings for a list of texts via the API with automatic batch splitting.
        """
        if not texts:
            return []

        batch_size = (
            self._initial_batch_size
            if self._initial_batch_size is not None
            else len(texts)
        )
        # Start with full batch, will be subdivided if needed
        return self._get_embeddings_with_retry(texts, initial_batch_size=batch_size)

    def _get_embeddings_with_retry(
        self, texts: List[str], initial_batch_size: int, max_retries: int = 3
    ) -> List[List[float]]:
        """
        Internal method to handle batch subdivision and retries.
        """
        batch_size = min(initial_batch_size, len(texts))
        total_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            for attempt in range(max_retries):
                try:
                    embeddings = self._single_api_call(batch_texts)
                    total_embeddings.extend(embeddings)
                    break  # Success, move to next batch

                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 400:
                        try:
                            error_response = e.response.json()
                            error_message = error_response.get("message", "").lower()

                            # Check if it's a batch size error
                            if any(
                                keyword in error_message
                                for keyword in [
                                    "too many inputs",
                                    "split into more batches",
                                    "batch size",
                                    "request too large",
                                ]
                            ):
                                # Reduce batch size by half
                                new_batch_size = max(1, len(batch_texts) // 2)
                                tqdm.write(
                                    f"🔄 Batch too large ({len(batch_texts)} texts), "
                                    f"splitting into smaller batches of {new_batch_size}"
                                )

                                # Recursively process with smaller batches
                                sub_embeddings = self._get_embeddings_with_retry(
                                    batch_texts, new_batch_size, max_retries
                                )
                                total_embeddings.extend(sub_embeddings)
                                break  # Success with subdivision
                            else:
                                # Other 400 error, don't retry
                                raise
                        except (json.JSONDecodeError, KeyError):
                            # Can't parse error response, don't retry
                            raise
                    else:
                        # Non-400 error, don't retry
                        raise

                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        # Last attempt failed
                        error_msg = f"❌ API Error after {max_retries} attempts: {e}"
                        if hasattr(e, "response") and e.response is not None:
                            error_msg += f"\n  Status code: {e.response.status_code}"
                            error_msg += (
                                f"\n  URL: {getattr(e.response, 'url', 'unknown')}"
                            )
                            error_msg += f"\n  Response content: {e.response.text}"
                        tqdm.write(error_msg)
                        return []

        return total_embeddings

    def _single_api_call(self, texts: List[str]) -> List[List[float]]:
        """
        Makes a single API call without retry logic.
        """
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model, "input": texts}

        try:
            # Make the API request
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            embeddings = [data["embedding"] for data in result["data"]]
        except requests.exceptions.RequestException as e:
            tqdm.write(f"❌ API request failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                tqdm.write(f"  Status code: {e.response.status_code}")
                tqdm.write(f"  URL: {getattr(e.response, 'url', 'unknown')}")
                tqdm.write(f"  Response content: {e.response.text}")
            # Also log the full response for debugging
            if hasattr(e, "response") and e.response is not None:
                try:
                    tqdm.write(f"  Full response JSON: {e.response.json()}")
                except json.JSONDecodeError:
                    tqdm.write("  Could not decode JSON from response.")
            return []  # Return empty embeddings

        if self.expected_dimension and embeddings:
            actual_dimension = len(embeddings[0])
            if actual_dimension != self.expected_dimension:
                raise ValueError(
                    f"Expected dimension {self.expected_dimension}, but got {actual_dimension} for model {self.model}"
                )

        return embeddings
