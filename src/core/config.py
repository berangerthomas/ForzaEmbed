"""Configuration management for ForzaEmbed.

This module defines Pydantic models for application configuration and provides
functions to load and validate YAML configuration files. It handles all
configuration aspects including grid search parameters, model settings,
database options, and multiprocessing settings.

Example:
    Load a configuration file::

        from src.core.config import load_config

        config = load_config("configs/config.yml")
        print(config.models_to_test)
"""

import logging
from typing import Dict, List

import yaml
from pydantic import BaseModel


class GridSearchParams(BaseModel):
    """Configuration for grid search parameters.

    Attributes:
        chunk_size: List of chunk sizes to test (in characters).
        chunk_overlap: List of chunk overlaps to test (in characters).
        chunking_strategy: List of chunking strategies to evaluate.
        similarity_metrics: List of similarity metrics to use.
        themes: Mapping of theme names to lists of theme keywords.
    """

    chunk_size: List[int]
    chunk_overlap: List[int]
    chunking_strategy: List[str]
    similarity_metrics: List[str]
    themes: Dict[str, List[str]]


class ModelConfig(BaseModel):
    """Configuration for an embedding model.

    Attributes:
        type: The type of model (e.g., 'api', 'fastembed', 'sentence_transformers').
        name: The model name or identifier.
        dimensions: The embedding dimension of the model.
        base_url: Optional base URL for API-based models.
        timeout: Optional request timeout in seconds for API models.
        max_tokens: Optional maximum number of tokens per text. When a text exceeds 
            this limit, it will be split into smaller chunks and recombined.
            If None, uses model default (typically 512).
        pooling_strategy: Optional strategy for combining chunk embeddings when text 
            exceeds max_tokens. Options: "max" (default), "average", "weighted", "last".
            - "max": Max pooling - captures most salient features
            - "average": Mean of all chunks - preserves overall semantics
            - "weighted": First chunks weighted more - useful for structured documents
            - "last": Uses only last chunk - useful for summaries/conclusions
    """

    type: str
    name: str
    dimensions: int
    base_url: str | None = None
    timeout: int | None = None
    max_tokens: int | None = None
    pooling_strategy: str = "max"


class DatabaseSettings(BaseModel):
    """Configuration for database settings.

    Attributes:
        intelligent_quantization: Whether to enable intelligent quantization
            for reducing storage size.
        quantize_metrics: Whether to quantize metrics (similarities, scores).
            If True, metrics are stored with reduced precision (uint16) to save space.
            If False, metrics are stored in full float32 precision.
            Set to False if you need exact metric values without any quantization loss.
    """

    intelligent_quantization: bool
    quantize_metrics: bool = True


class EmbeddingPoolingStrategy(str):
    """Strategy for combining embeddings when text exceeds model token limit.
    
    When a text is too long for the embedding model, it's split into smaller
    chunks and their embeddings are combined using one of these strategies:
    
    - "max": Max pooling - takes the maximum value across all chunks for each 
             dimension. Best for capturing the most salient features.
    - "average": Average pooling - computes the mean of all chunk embeddings.
                 Preserves overall semantic content but may dilute important features.
    - "weighted": Weighted pooling - gives more importance to the first chunks.
                  Useful when the beginning of text is more informative.
    - "last": Uses only the last chunk embedding. Useful when the end of text 
              contains summaries or conclusions.
    """
    
    MAX = "max"
    AVERAGE = "average"
    WEIGHTED = "weighted"
    LAST = "last"


class MultiprocessingSettings(BaseModel):
    """Configuration for multiprocessing settings.

    Attributes:
        max_workers_api: Maximum number of workers for API-based embedding calls.
        max_workers_local: Optional maximum workers for local model inference.
        maxtasksperchild: Maximum tasks per worker child before respawning.
        embedding_batch_size_api: Batch size for API embedding requests.
        embedding_batch_size_local: Batch size for local model embedding.
        file_batch_size: Number of files to process per batch.
        api_batch_sizes: Mapping of provider names to their specific batch sizes.
    """

    max_workers_api: int
    max_workers_local: int | None = None
    maxtasksperchild: int
    embedding_batch_size_api: int
    embedding_batch_size_local: int
    file_batch_size: int
    api_batch_sizes: Dict[str, int]


class AppConfig(BaseModel):
    """Main application configuration.

    Attributes:
        grid_search_params: Configuration for grid search parameters.
        models_to_test: List of model configurations to evaluate.
        similarity_threshold: Threshold for similarity-based filtering.
        output_dir: Directory path for output files.
        generate_filtered_markdowns: Whether to generate filtered markdown files.
        database: Database-related settings.
        multiprocessing: Multiprocessing-related settings.
    """

    grid_search_params: GridSearchParams
    models_to_test: List[ModelConfig]
    similarity_threshold: float
    output_dir: str
    generate_filtered_markdowns: bool = False
    database: DatabaseSettings
    multiprocessing: MultiprocessingSettings


def load_config(config_path: str) -> AppConfig:
    """Load and validate a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A validated AppConfig instance.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML file is malformed.
        pydantic.ValidationError: If the configuration fails validation.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        return AppConfig(**config_data)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logging.error(f"Error validating configuration: {e}")
        raise
