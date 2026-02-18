"""Data aggregation module for ForzaEmbed reporting.

This module provides the DataAggregator class that handles aggregation
and caching of processed data from the database for report generation.

Example:
    Aggregate data for reporting::

        from src.reporting.aggregator import DataAggregator

        aggregator = DataAggregator(db, output_dir, "config_name")
        data = aggregator.get_aggregated_data()
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from tqdm import tqdm

from ..utils.database import EmbeddingDatabase


class DataAggregator:
    """Handle aggregation and caching of processed data from the database.

    Aggregates processing results from the database into a format suitable
    for report generation, with caching to avoid redundant computation.

    Attributes:
        db: The embedding database containing results.
        output_dir: Directory path for cache files.
        cache_path: Path to the cache file.
    """

    def __init__(
        self, db: EmbeddingDatabase, output_dir: Path, config_name: str
    ) -> None:
        """Initialize the DataAggregator.

        Args:
            db: The embedding database containing results.
            output_dir: Directory path for cache files.
            config_name: Name of the configuration for cache file prefix.
        """
        self.db = db
        self.output_dir = output_dir
        self.cache_path = self.output_dir / f"{config_name}_reports_cache.joblib"

    def get_aggregated_data(self) -> dict[str, Any] | None:
        """Load aggregated data from cache if valid, otherwise aggregate from scratch.

        Checks if the cache is newer than the database modification time.
        If valid, loads from cache; otherwise, aggregates fresh data.

        Returns:
            Dictionary containing aggregated data for reporting, or None if
            no processing results are available. Contains keys:
                - all_results: Raw results from database.
                - processed_data_for_interactive_page: Optimized web data.
                - all_models_metrics: Metrics organized by model.
                - model_embeddings_for_variance: Embeddings for analysis.
                - total_combinations: Count of model combinations.
        """
        db_mod_time = self.db.get_db_modification_time()
        use_cache = (
            self.cache_path.exists() and self.cache_path.stat().st_mtime > db_mod_time
        )

        if use_cache:
            logging.info(f"Loading aggregated data from cache: {self.cache_path}")
            return joblib.load(self.cache_path)

        logging.info("No valid cache found. Aggregating data from scratch...")
        all_results = self.db.get_all_processing_results()
        if not all_results:
            logging.warning("No processing results found in the database.")
            return None

        aggregated_data = self._aggregate_data(all_results)
        joblib.dump(aggregated_data, self.cache_path)
        logging.info(f"Saved aggregated data to cache: {self.cache_path}")
        return aggregated_data

    def _aggregate_data(self, all_results: dict[str, Any]) -> dict[str, Any]:
        """Aggregate data from results for reporting.

        Args:
            all_results: Dictionary of processing results from database.

        Returns:
            Dictionary containing aggregated and processed data.
        """
        processed_data_for_interactive_page: dict[str, Any] = {"files": {}}
        all_models_metrics: dict[str, list[dict[str, Any]]] = {}
        model_embeddings_for_variance: dict[str, dict[str, Any]] = {}

        # First, normalize dot product scores globally if needed
        all_results = self._normalize_dot_product_globally(all_results)

        for model_name, model_results in tqdm(
            all_results.items(), desc="Aggregating data for reports"
        ):
            # Aggregate embeddings and labels from all files for this model
            aggregated_embeddings: list[Any] = []
            aggregated_labels: list[Any] = []
            for file_id, file_data in model_results.get("files", {}).items():
                if "embeddings" in file_data and file_data["embeddings"] is not None:
                    aggregated_embeddings.extend(file_data["embeddings"])
                if "labels" in file_data and file_data["labels"] is not None:
                    aggregated_labels.extend(file_data["labels"])

            model_embeddings_for_variance[model_name] = {
                "embeddings": np.array(aggregated_embeddings)
                if aggregated_embeddings
                else np.array([]),
                "labels": aggregated_labels,
            }

            # Prepare data for the interactive page
            for file_id, file_data in model_results.get("files", {}).items():
                file_name = file_data.get("file_name", file_id)
                file_entry = processed_data_for_interactive_page["files"].setdefault(
                    file_id, {"fileName": file_name, "embeddings": {}}
                )
                file_entry["embeddings"][model_name] = {
                    "phrases": file_data.get("phrases", []),
                    "similarities": file_data.get("similarities", []),
                    "metrics": file_data.get("metrics", {}),
                    "scatter_plot_data": file_data.get("scatter_plot_data"),
                }

            # Store detailed metrics for each file
            detailed_metrics: list[dict[str, Any]] = []
            for file_id, file_data in model_results.get("files", {}).items():
                if "metrics" in file_data and file_data["metrics"]:
                    metric_record: dict[str, Any] = {"file_name": file_id}
                    metric_record.update(file_data["metrics"])
                    detailed_metrics.append(metric_record)
            all_models_metrics[model_name] = detailed_metrics

        optimized_data = self._optimize_data_for_web(
            processed_data_for_interactive_page
        )

        return {
            "all_results": all_results,
            "processed_data_for_interactive_page": optimized_data,
            "all_models_metrics": all_models_metrics,
            "model_embeddings_for_variance": model_embeddings_for_variance,
            "total_combinations": len(all_results),
        }

    def _optimize_data_for_web(self, data: dict[str, Any]) -> dict[str, Any]:
        """Optimize the data structure for web output by rounding floats.

        Args:
            data: Data dictionary to optimize.

        Returns:
            Optimized data dictionary with floats rounded to 4 decimal places.
        """
        from typing import cast

        def round_floats(obj: Any) -> Any:
            if isinstance(obj, list):
                return [round_floats(v) for v in obj]
            if isinstance(obj, dict):
                return {k: round_floats(v) for k, v in obj.items()}
            if isinstance(obj, float):
                return round(obj, 4)
            return obj

        return cast(dict[str, Any], round_floats(data))

    def _normalize_dot_product_globally(
        self, all_results: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform global min-max scaling on dot product similarities.

        Args:
            all_results: Dictionary of processing results.

        Returns:
            Results with dot product similarities normalized to [0, 1].
        """
        for model_name, model_results in all_results.items():
            model_info = self.db.get_model_info(model_name)
            if not model_info or model_info.get("similarity_metric") != "dot_product":
                continue

            all_similarities: list[float] = [
                sim
                for file_data in model_results.get("files", {}).values()
                if file_data.get("similarities") is not None
                for sim in file_data["similarities"]
            ]

            if not all_similarities:
                continue

            dot_min, dot_max = min(all_similarities), max(all_similarities)
            denominator = dot_max - dot_min

            if denominator == 0:
                for file_data in model_results.get("files", {}).values():
                    if "similarities" in file_data:
                        file_data["similarities"] = [0.5] * len(
                            file_data["similarities"]
                        )
                continue

            for file_data in model_results.get("files", {}).values():
                if "similarities" in file_data:
                    file_data["similarities"] = [
                        (s - dot_min) / denominator for s in file_data["similarities"]
                    ]

        return all_results

    def touch_cache(self) -> None:
        """Update the cache file's modification time to the current time."""
        if self.cache_path.exists():
            self.cache_path.touch()
            logging.info(f"Updated cache timestamp: {self.cache_path}")
