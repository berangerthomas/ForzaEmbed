"""API client for production embedding services.

This module provides a client for obtaining embeddings from production APIs
including OpenAI, Mistral, and VoyageAI. It handles authentication, batching,
and automatic retry with batch size reduction on errors.

Example:
    Get embeddings from an API::

        from src.clients.api_client import ProductionEmbeddingClient

        client = ProductionEmbeddingClient(
            base_url="https://api.openai.com/v1",
            model="text-embedding-ada-002",
            expected_dimension=1536
        )
        embeddings = client.get_embeddings(["Hello", "World"])
"""

import json
import os
from typing import List

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


class ProductionEmbeddingClient:
    """Client for obtaining embeddings from production APIs.

    Supports OpenAI-compatible APIs with automatic API key selection based
    on the model name. Implements automatic batch splitting and retries.

    Attributes:
        base_url: Base URL of the API.
        model: Name of the embedding model.
        expected_dimension: Expected embedding dimension for validation.
        timeout: Request timeout in seconds.
        session: Requests session with authentication headers.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        expected_dimension: int | None = None,
        timeout: int = 30,
        initial_batch_size: int | None = None,
    ) -> None:
        """Initialize the ProductionEmbeddingClient.

        Args:
            base_url: Base URL of the API.
            model: Name of the embedding model to use.
            expected_dimension: Expected dimension of embeddings for validation.
            timeout: Timeout for requests in seconds.
            initial_batch_size: Initial batch size for requests.
        """
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

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Retrieve embeddings for a list of texts via the API.

        Implements automatic batch splitting for large requests.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors as lists of floats.
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
        """Handle batch subdivision and retries for embedding requests.

        Automatically reduces batch size on 400 errors related to batch limits.

        Args:
            texts: List of texts to embed.
            initial_batch_size: Starting batch size.
            max_retries: Maximum number of retry attempts.

        Returns:
            List of embedding vectors.
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
        """Make a single API call without retry logic.

        Args:
            texts: List of texts to embed in this call.

        Returns:
            List of embedding vectors.

        Raises:
            requests.exceptions.HTTPError: On API errors.
            ValueError: If embedding dimension doesn't match expected.
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
