import logging
import textwrap
from pathlib import Path
from typing import Any, Dict

import joblib
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

    def generate_all(self, top_n: int | None = None, single_file: bool = False):
        """
        Generates all reports from the data in the database.
        """
        logging.info("--- Generating All Reports ---")
        cache_path = self.output_dir / "reports_cache.joblib"
        db_mod_time = self.db.get_db_modification_time()

        use_cache = cache_path.exists() and cache_path.stat().st_mtime > db_mod_time
        
        all_results = {}
        aggregated_data = None

        if use_cache:
            logging.info(f"Loading aggregated data from cache: {cache_path}")
            aggregated_data = joblib.load(cache_path)
        else:
            logging.info("No valid cache found. Aggregating data from scratch...")
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
            total_combinations = len(all_results)
            aggregated_data = (
                processed_data_for_interactive_page,
                all_models_metrics,
                model_embeddings_for_variance,
                total_combinations,
            )

        (
            processed_data_for_interactive_page,
            all_models_metrics,
            model_embeddings_for_variance,
            total_combinations,
        ) = aggregated_data

        self._generate_main_web_page(
            processed_data_for_interactive_page, total_combinations, single_file
        )
        self._generate_file_reports(all_results)
        self._generate_global_reports(
            all_models_metrics, model_embeddings_for_variance, top_n
        )

        # After all reports are generated (including DB writes), save the cache
        if not use_cache and aggregated_data:
            joblib.dump(aggregated_data, cache_path)
            logging.info(f"Saved aggregated data to cache: {cache_path}")

        logging.info(f"All reports generated in '{self.output_dir}'.")

    def _aggregate_data(self, all_results: dict):
        """Aggregates data from results for reporting."""
        processed_data_for_interactive_page = {"files": {}}
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
                file_entry = processed_data_for_interactive_page["files"].setdefault(
                    file_id, {"embeddings": {}}
                )
                file_entry["embeddings"][model_name] = {
                    "phrases": file_data.get("phrases", []),
                    "similarities": file_data.get("similarities", []),
                    "metrics": file_data.get("metrics", {}),
                    "scatter_plot_data": file_data.get("scatter_plot_data"),
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

        optimized_data = self._optimize_data_for_web(
            processed_data_for_interactive_page
        )

        return (
            optimized_data,
            all_models_metrics,
            model_embeddings_for_variance,
        )

    def _optimize_data_for_web(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimizes the data structure for web output by rounding floats.
        """

        def round_floats(obj):
            if isinstance(obj, list):
                return [round_floats(v) for v in obj]
            if isinstance(obj, dict):
                return {k: round_floats(v) for k, v in obj.items()}
            if isinstance(obj, float):
                return round(obj, 4)
            return obj

        return round_floats(data)

    def _generate_main_web_page(
        self, processed_data, total_combinations, single_file: bool = False
    ):
        """Generates the main interactive web page."""
        from .web_generator import generate_main_page

        generate_main_page(
            processed_data,
            str(self.output_dir),
            total_combinations,
            single_file=single_file,
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
            plot_paths = self._analyze_and_visualize_clustering_metrics(
                all_models_metrics, top_n=top_n
            )
            if plot_paths:
                for path in plot_paths:
                    chart_name = path.stem
                    self.db.add_global_chart(chart_name, str(path))

    def _plot_single_metric(
        self,
        df: pd.DataFrame,
        metric: str,
        output_path: Path,
        higher_is_better: bool,
        top_n: int | None = None,
    ) -> None:
        """Generates and saves a sorted bar plot for a single metric."""
        sorted_df = df.sort_values(by=metric, ascending=not higher_is_better)

        if top_n:
            sorted_df = sorted_df.head(top_n)

        plt.figure(figsize=(18, 12))
        ax = sns.barplot(
            x=sorted_df.index,
            y=sorted_df[metric],
            palette="viridis",
            hue=sorted_df.index,
            legend=False,
        )

        title_suffix = "(Higher is Better)" if higher_is_better else "(Lower is Better)"
        ax.set_title(
            f"Model Comparison - {metric.replace('_', ' ').title()} {title_suffix}",
            pad=20,
            fontsize=18,
        )
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=14)
        ax.set_xlabel("Model", fontsize=14)

        # Wrap labels
        labels = [
            textwrap.fill(label, width=30, break_long_words=False)
            for label in sorted_df.index
        ]
        ax.set_xticks(ax.get_xticks())  # Explicitly set tick locations
        ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")

        plt.tight_layout(pad=3.0)
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Saved {metric} plot to {output_path}")

    def _generate_radar_chart(self, df: pd.DataFrame) -> Path | None:
        """Generates a radar chart for the most important metrics."""
        metrics_for_radar = {
            "discriminant_score": True,
            "silhouette": True,
            "cohesion": False,
            "separation": True,
        }

        plot_metrics = [m for m in metrics_for_radar if m in df.columns]
        if len(plot_metrics) < 3:
            logging.warning("Not enough metrics for a radar chart.")
            return None

        # Normalize the data
        normalized_df = df[plot_metrics].copy()
        for metric, higher_is_better in metrics_for_radar.items():
            if metric in normalized_df.columns:
                min_val = normalized_df[metric].min()
                max_val = normalized_df[metric].max()
                if max_val - min_val > 0:
                    normalized_df[metric] = (normalized_df[metric] - min_val) / (
                        max_val - min_val
                    )
                    if not higher_is_better:
                        normalized_df[metric] = 1 - normalized_df[metric]
                else:
                    normalized_df[metric] = 0.5  # Neutral if all values are the same

        # Plotting
        labels = normalized_df.columns
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

        for i, row in normalized_df.iterrows():
            values = row.tolist()
            values += values[:1]
            ax.plot(angles, values, label=textwrap.fill(str(i), 20))
            ax.fill(angles, values, alpha=0.1)

        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

        plt.title("Key Metrics Radar Chart", size=20, y=1.1)

        radar_path = self.output_dir / "global_radar_chart.png"
        plt.savefig(radar_path, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved radar chart to {radar_path}")
        return radar_path

    def _analyze_and_visualize_clustering_metrics(
        self, all_models_metrics: Dict[str, Dict[str, float]], top_n: int | None = None
    ) -> list[Path]:
        """
        Analyzes clustering metrics, visualizes each in a separate plot,
        and generates a summary radar chart.
        Returns a list of paths to the generated plots.
        """
        if not all_models_metrics:
            return []

        df = pd.DataFrame.from_dict(all_models_metrics, orient="index").dropna()
        if df.empty:
            return []

        if "discriminant_score" in df.columns:
            df = df.sort_values(by="discriminant_score", ascending=False)

        # Export metrics to CSV
        csv_path = self.output_dir / "global_metrics_comparison.csv"
        df.to_csv(csv_path)
        logging.info(f"Exported global metrics to {csv_path}")

        if top_n:
            df = df.head(top_n)

        metric_preferences = {
            "cohesion": False,
            "separation": True,
            "discriminant_score": True,
            "silhouette": True,
        }

        metrics_to_plot = [m for m in metric_preferences if m in df.columns]
        if not metrics_to_plot:
            return []

        plot_paths = []
        for metric in metrics_to_plot:
            plot_path = self.output_dir / f"global_{metric}_comparison.png"
            self._plot_single_metric(
                df,
                metric,
                plot_path,
                higher_is_better=metric_preferences[metric],
                top_n=top_n,
            )
            plot_paths.append(plot_path)

        # Generate and add radar chart
        radar_path = self._generate_radar_chart(df)
        if radar_path:
            plot_paths.append(radar_path)

        return plot_paths
