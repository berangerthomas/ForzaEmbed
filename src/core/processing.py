"""Processing module for embedding analysis.

This module contains the Processor class that handles the core data processing
logic for embedding generation, similarity calculation, and metric evaluation.
It coordinates specialized services to execute individual test runs.

Example:
    Process a test run::

        from src.core.processing import Processor

        processor = Processor(db, config)
        result = processor.run_test(rows, model_config, ...)
"""

import hashlib
import logging
from typing import Any

import numpy as np
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from tqdm import tqdm

from ..metrics.evaluation_metrics import calculate_all_metrics
from ..services.embedding_service import EmbeddingService
from ..services.similarity_service import SimilarityService
from ..services.visualization_service import VisualizationService
from ..utils.database import EmbeddingDatabase
from ..utils.utils import chunk_text
from .config import AppConfig, ModelConfig


class Processor:
    """Handle core data processing logic for embedding analysis.

    This class orchestrates the processing pipeline for a single test run,
    delegating specific tasks to specialized services for embedding generation,
    similarity calculation, and visualization.

    Attributes:
        db: The embedding database instance.
        config: The application configuration.
        embedding_service: Service for embedding generation and caching.
        similarity_service: Service for similarity calculations.
        visualization_service: Service for t-SNE visualization.
    """

    def __init__(self, db: EmbeddingDatabase, config: AppConfig) -> None:
        """Initialize the Processor.

        Args:
            db: The embedding database instance for storing results.
            config: The application configuration.
        """
        self.db = db
        self.config = config
        self.embedding_service = EmbeddingService(db, config)
        self.similarity_service = SimilarityService()
        self.visualization_service = VisualizationService(db)

    def _safe_convert_to_python_types(self, data: Any) -> Any:
        """Recursively convert NumPy types to native Python types.

        Ensures data can be serialized to JSON by converting numpy arrays,
        floats, and integers to their Python equivalents.

        Args:
            data: The data to convert, which may contain NumPy types.

        Returns:
            The data with all NumPy types converted to Python native types.
        """
        if isinstance(data, np.ndarray):
            return data.astype(float).tolist()
        elif isinstance(data, (np.floating, float)):
            return float(data)
        elif isinstance(data, (np.integer, int)):
            return int(data)
        elif isinstance(data, dict):
            return {
                key: self._safe_convert_to_python_types(value)
                for key, value in data.items()
            }
        elif isinstance(data, (list, tuple)):
            return [self._safe_convert_to_python_types(item) for item in data]
        else:
            return data

    def run_test(
        self,
        rows: list[tuple[str, str]],
        model_config: ModelConfig,
        chunk_size: int,
        chunk_overlap: int,
        themes: list[str],
        theme_name: str,
        chunking_strategy: str,
        similarity_metric: str,
        processed_files: list[str],
        pbar: tqdm,
    ) -> dict[str, Any]:
        """Process a test run for a specific parameter combination.

        Handles the complete workflow for processing files including embedding
        generation, similarity calculation, and metric evaluation.

        Args:
            rows: List of (name, content) tuples for files to process.
            model_config: The model configuration to use.
            chunk_size: Size of text chunks in characters.
            chunk_overlap: Overlap between chunks in characters.
            themes: List of theme keywords to compare against.
            theme_name: Name identifier for the theme set.
            chunking_strategy: The chunking strategy to use.
            similarity_metric: The similarity metric to use.
            processed_files: List of file names already processed.
            pbar: Progress bar object for status updates.

        Returns:
            Dictionary containing processing results with file data and metrics.
        """
        model_name = model_config.name
        results = {"files": {}}

        embedding_function = self.embedding_service.get_embedding_function(model_config)
        # Use the base model name for caching, as embeddings are model-specific
        base_model_name = model_config.name

        themes_embeddings_map, themes_time = self.embedding_service.get_or_create_embeddings(
            embedding_function, base_model_name, themes
        )

        unprocessed_rows = [row for row in rows if row[0] not in processed_files]

        if not themes_embeddings_map:
            logging.error(
                f"Failed to embed themes for model '{model_name}'. "
                f"Skipping {len(unprocessed_rows)} files for this combination."
            )
            pbar.update(len(unprocessed_rows))
            return {"results": {}}

        theme_hashes = [self.embedding_service.get_text_hash(theme) for theme in themes]
        embed_themes_list = [
            themes_embeddings_map[h] for h in theme_hashes if h in themes_embeddings_map
        ]
        embed_themes = np.array(embed_themes_list)

        for item in unprocessed_rows:
            name, texte = item
            grid_params = (
                f"cs{chunk_size} co{chunk_overlap} {similarity_metric[:3]} "
                f"{chunking_strategy} {theme_name[-13:]}"
            )
            description = f"{name} | {model_name[-20:]} | {grid_params}"
            pbar.set_description(description[:150])
            if not texte or not texte.strip():
                pbar.update(1)
                continue

            try:
                language = detect(texte)
            except LangDetectException:
                language = "en"  # Default to English if detection fails

            item_phrases = chunk_text(
                texte, chunk_size, chunk_overlap, chunking_strategy, language=language
            )
            if not item_phrases:
                pbar.update(1)
                continue

            all_embeddings_map, chunks_time = self.embedding_service.get_or_create_embeddings(
                embedding_function, base_model_name, item_phrases
            )

            phrase_hashes = {p: self.embedding_service.get_text_hash(p) for p in item_phrases}
            item_embed_phrases = np.array(
                [
                    all_embeddings_map[h]
                    for p in item_phrases
                    if (h := phrase_hashes[p]) in all_embeddings_map
                ]
            )

            if item_embed_phrases.size == 0:
                pbar.update(1)
                continue

            if embed_themes.shape[1] != item_embed_phrases.shape[1]:
                logging.error(f"Dimension mismatch for file '{name}'.")
                pbar.update(1)
                continue

            # Calculate similarities and labels with validation
            similarites = self.similarity_service.calculate_similarity(
                embed_themes, item_embed_phrases, similarity_metric
            )
            similarites = self.similarity_service.validate_similarities(similarites, similarity_metric)
            labels = np.argmax(similarites, axis=0)

            # Calculate metrics
            all_metrics = calculate_all_metrics(
                embed_themes,
                item_embed_phrases,
                labels,
            )

            # Add timing metrics
            total_embedding_time = themes_time + chunks_time
            all_metrics["embedding_computation_time"] = float(total_embedding_time)

            # Convert metrics to safe Python types
            all_metrics = self._safe_convert_to_python_types(all_metrics)

            # Generate t-SNE coordinates
            tsne_key = f"{model_config.name}_cs{chunk_size}_co{chunk_overlap}_{chunking_strategy}"
            scatter_plot_data = self.visualization_service.get_or_create_tsne_data(
                item_embed_phrases,
                tsne_key,
                name,
                similarites,
                self.config.similarity_threshold,
            )

            # Ensure all data is properly converted
            max_similarities = similarites.max(axis=0)
            safe_similarities = self._safe_convert_to_python_types(max_similarities)

            results["files"][name] = {
                "file_name": name,
                "phrases": item_phrases,
                "similarities": safe_similarities,
                "metrics": all_metrics,
                "scatter_plot_data": scatter_plot_data,
            }
            pbar.update(1)

        return {"results": results}
