import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from .database import EmbeddingDatabase


class ReportGenerator:
    """
    Handles the generation of all reports and visualizations.
    """

    def __init__(self, db: EmbeddingDatabase, config: Dict[str, Any], output_dir: Path):
        self.db = db
        self.config = config
        self.output_dir = output_dir
        self.similarity_threshold = config.get("similarity_threshold", 0.6)
        self.file_metadata = {}  # To be loaded when needed

    def generate_all(self, top_n: int | None = None):
        """
        Generates all reports from the data in the database.
        """
        logging.info("--- Generating All Reports ---")

        all_results = self.db.get_all_processing_results()
        if not all_results:
            logging.warning(
                "No processing results found in the database. Run processing first."
            )
            return

        (
            processed_data_for_interactive_page,
            all_models_metrics,
            model_embeddings_for_variance,
        ) = self._aggregate_data(all_results)

        # Placeholder for web generator
        # self._generate_main_web_page(processed_data_for_interactive_page, len(all_results))

        self._generate_file_reports(all_results)
        self._generate_global_reports(
            all_models_metrics, model_embeddings_for_variance, top_n
        )

        logging.info(f"All reports generated in '{self.output_dir}'.")

    def _aggregate_data(self, all_results: dict):
        """Aggregates data from results for reporting."""
        processed_data_for_interactive_page = {}
        all_models_metrics = {}
        model_embeddings_for_variance = {}

        for model_name, model_results in tqdm(
            all_results.items(), desc="Aggregating data for reports"
        ):
            model_info = self.db.get_model_info(model_name)
            if not model_info:
                continue
            base_model_name = model_info["model_name"]

            for file_id, file_data in model_results.get("files", {}).items():
                file_entry = processed_data_for_interactive_page.setdefault(
                    file_id, {"embeddings": {}}
                )
                file_entry["embeddings"][model_name] = {
                    "phrases": file_data.get("phrases", []),
                    "similarities": file_data.get("similarities", []),
                    "metrics": file_data.get("metrics", {}),
                }

            metrics_list = [
                res["metrics"]
                for res in model_results.get("files", {}).values()
                if res and "metrics" in res
            ]
            if metrics_list:
                avg_metrics = {
                    key: float(np.mean([m[key] for m in metrics_list if key in m]))
                    for key in metrics_list[0]
                }
                all_models_metrics[model_name] = avg_metrics
                self.db.add_evaluation_metrics(model_name, avg_metrics)

        return (
            processed_data_for_interactive_page,
            all_models_metrics,
            model_embeddings_for_variance,
        )

    def _generate_file_reports(self, all_results):
        """Generates individual Markdown reports for each file."""
        logging.info("Generating file-specific reports...")
        # This part will be implemented in detail later
        pass

    def _generate_global_reports(
        self, all_models_metrics, model_embeddings_for_variance, top_n=None
    ):
        """Generates global comparison charts."""
        logging.info("Generating global reports...")
        if all_models_metrics:
            plot_path = self._analyze_and_visualize_clustering_metrics(
                all_models_metrics, top_n=top_n
            )
            if plot_path:
                self.db.add_global_chart("clustering_metrics", str(plot_path))

    def _analyze_and_visualize_clustering_metrics(
        self, all_models_metrics: Dict[str, Dict[str, float]], top_n: int | None = None
    ) -> Path | None:
        """Analyzes and visualizes clustering metrics for all models."""
        if not all_models_metrics:
            return None

        df = pd.DataFrame.from_dict(all_models_metrics, orient="index").dropna()
        if df.empty:
            return None

        if "discriminant_score" in df.columns:
            df = df.sort_values(by="discriminant_score", ascending=False)

        # Export metrics to CSV
        csv_path = self.output_dir / "global_metrics_comparison.csv"
        df.to_csv(csv_path)
        logging.info(f"Exported global metrics to {csv_path}")

        if top_n:
            df = df.head(top_n)

        metrics_to_plot = [
            "cohesion",
            "separation",
            "discriminant_score",
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
        ]
        metrics_to_plot = [m for m in metrics_to_plot if m in df.columns]
        if not metrics_to_plot:
            return None

        num_metrics = len(metrics_to_plot)
        plt.figure(figsize=(18, 5 * num_metrics))

        for i, metric in enumerate(metrics_to_plot, 1):
            plt.subplot(num_metrics, 1, i)
            sns.barplot(x=df.index, y=df[metric], palette="viridis")
            plt.title(f"Model Comparison - {metric.replace('_', ' ').title()}")
            plt.ylabel(metric.replace("_", " ").title())
            plt.xlabel("Model")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

        plot_path = self.output_dir / "global_clustering_metrics_comparison.png"
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved clustering metrics plot to {plot_path}")
        return plot_path
