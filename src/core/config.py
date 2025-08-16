import logging
import yaml
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class GridSearchParams(BaseModel):
    chunk_size: List[int]
    chunk_overlap: List[int]
    chunking_strategy: List[str]
    similarity_metrics: List[str]
    themes: Dict[str, List[str]]

class ModelConfig(BaseModel):
    type: str
    name: str
    dimensions: int
    base_url: str | None = None
    timeout: int | None = None

class DatabaseSettings(BaseModel):
    intelligent_quantization: bool

class MultiprocessingSettings(BaseModel):
    max_workers_api: int
    max_workers_local: int | None = None
    maxtasksperchild: int
    embedding_batch_size_api: int
    embedding_batch_size_local: int
    file_batch_size: int
    api_batch_sizes: Dict[str, int]

class AppConfig(BaseModel):
    grid_search_params: GridSearchParams
    models_to_test: List[ModelConfig]
    similarity_threshold: float
    output_dir: str
    database: DatabaseSettings
    multiprocessing: MultiprocessingSettings

def load_config(config_path: str) -> AppConfig:
    """Loads and validates the YAML configuration file."""
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
