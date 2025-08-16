import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils.database import EmbeddingDatabase


class DataAggregator:
    """
    Handles the aggregation and caching of processed data from the database.
    """

    def __init__(self, db: EmbeddingDatabase, output_dir: Path):
        self.db = db
        self.output_dir = output_dir
        self.cache_path = self.output_dir / "reports_cache.joblib"

    def get_aggregated_data(self) -> Dict[str, Any] | None:
        """
        Loads aggregated data from cache if valid, otherwise aggregates from scratch.
        """
        db_mod_time = self.db.get_db_modification_time()
        use_cache = self.cache_path.exists() and self.cache_path.stat().st_mtime > db_mod_time

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

    def _aggregate_data(self, all_results: dict) -> Dict[str, Any]:
        """Aggregates data from results for reporting."""
        processed_data_for_interactive_page = {"files": {}}
        all_models_metrics = {}
        model_embeddings_for_variance = {}

        for model_name, model_results in tqdm(
            all_results.items(), desc="Aggregating data for reports"
        ):
            # Aggregate embeddings and labels from all files for this model
            aggregated_embeddings = []
            aggregated_labels = []
            for file_data in model_results.get("files", {}).values():
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
            detailed_metrics = []
            for file_id, file_data in model_results.get("files", {}).items():
                if "metrics" in file_data and file_data["metrics"]:
                    metric_record = {"file_name": file_id}
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

    def _optimize_data_for_web(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimizes the data structure for web output by rounding floats.
        """
        from typing import cast

        def round_floats(obj):
            if isinstance(obj, list):
                return [round_floats(v) for v in obj]
            if isinstance(obj, dict):
                return {k: round_floats(v) for k, v in obj.items()}
            if isinstance(obj, float):
                return round(obj, 4)
            return obj

        return cast(Dict[str, Any], round_floats(data))

    def touch_cache(self):
        """Updates the cache file's modification time to the current time."""
        if self.cache_path.exists():
            self.cache_path.touch()
            logging.info(f"Updated cache timestamp: {self.cache_path}")
