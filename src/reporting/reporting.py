import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from ..utils.database import EmbeddingDatabase
from .aggregator import DataAggregator


class ReportGenerator:
    """
    Handles the generation of all reports and visualizations.
    """

    def __init__(
        self,
        db: EmbeddingDatabase,
        config: Dict[str, Any],
        output_dir: Path,
        config_name: str,
    ):
        self.db = db
        self.config = config
        self.output_dir = output_dir
        self.similarity_threshold = config.get("similarity_threshold", 0.6)
        self.data_aggregator = DataAggregator(db, output_dir, config_name)

    def generate_all(self, top_n: int = 25, single_file: bool = False):
        """
        Generates all reports from the data in the database.
        """
        logging.info("--- Generating All Reports ---")
        effective_top_n = None if top_n == -1 else top_n
        aggregated_data = self.data_aggregator.get_aggregated_data()

        if not aggregated_data:
            logging.warning("No aggregated data available. Skipping report generation.")
            return

        processed_data_for_interactive_page = aggregated_data[
            "processed_data_for_interactive_page"
        ]
        total_combinations = aggregated_data["total_combinations"]
        all_results = aggregated_data["all_results"]

        graph_paths_by_file: Dict[str, Any] = {}

        if single_file:
            all_models_metrics = aggregated_data["all_models_metrics"]
            global_plot_paths = self._generate_global_reports(
                all_models_metrics, effective_top_n
            )
            graph_paths_by_file["global"] = global_plot_paths
        else:
            for file_id in processed_data_for_interactive_page["files"]:
                file_specific_metrics = {}
                for model_name, model_data in all_results.items():
                    if file_id in model_data.get("files", {}):
                        metrics = model_data["files"][file_id].get("metrics")
                        if metrics:
                            metric_record = {"file_name": file_id}
                            metric_record.update(metrics)
                            file_specific_metrics[model_name] = [metric_record]

                if file_specific_metrics:
                    file_prefix = Path(file_id).stem
                    plot_paths = self._generate_global_reports(
                        file_specific_metrics,
                        top_n=effective_top_n,
                        file_prefix=file_prefix,
                    )
                    graph_paths_by_file[file_id] = plot_paths

        self._generate_main_web_page(
            processed_data_for_interactive_page,
            total_combinations,
            single_file,
            graph_paths_by_file,
        )

        logging.info(f"All reports generated in '{self.output_dir}'.")
        self.data_aggregator.touch_cache()

    def _generate_main_web_page(
        self,
        processed_data,
        total_combinations,
        single_file: bool = False,
        graph_paths: Dict[str, Any] | None = None,
    ):
        """Generates the main interactive web page."""
        from .web_generator import generate_main_page

        generate_main_page(
            processed_data,
            str(self.output_dir),
            total_combinations,
            single_file=single_file,
            graph_paths=graph_paths,
        )

    def _generate_global_reports(
        self, all_models_metrics, top_n=None, file_prefix: str = "global"
    ):
        """Generates global comparison charts."""
        logging.info(f"Generating reports for prefix: {file_prefix}...")
        if all_models_metrics:
            plot_paths = self._analyze_and_visualize_clustering_metrics(
                all_models_metrics, top_n=top_n, file_prefix=file_prefix
            )
            if plot_paths and file_prefix == "global":
                for path in plot_paths:
                    chart_name = path.stem
                    self.db.add_global_chart(chart_name, str(path))
            return plot_paths
        return []

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

    def _generate_radar_chart(
        self, df: pd.DataFrame, file_prefix: str = "global"
    ) -> Path | None:
        """Generates a radar chart for the most important metrics."""
        metrics_for_radar = {
            "silhouette_score": True,
            "inter_cluster_distance_normalized": True,
            "intra_cluster_distance_normalized": True,
            "internal_coherence_score": False,
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

        radar_path = self.output_dir / f"{file_prefix}_radar_chart.png"
        plt.savefig(radar_path, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved radar chart to {radar_path}")
        return radar_path

    def _analyze_and_visualize_clustering_metrics(
        self,
        all_models_metrics: Dict[str, Any],
        top_n: int | None = None,
        file_prefix: str = "global",
    ) -> list[Path]:
        """
        Analyzes clustering metrics, visualizes each in a separate plot,
        and generates a summary radar chart.
        Returns a list of paths to the generated plots.
        """
        if not all_models_metrics:
            return []

        # Convert the new structure (dict of lists of dicts) to a DataFrame
        records = []
        for model_name, metrics_list in all_models_metrics.items():
            for metric_record in metrics_list:
                record = {"model_name": model_name}
                record.update(metric_record)
                records.append(record)

        if not records:
            return []

        df = pd.DataFrame(records)
        if df.empty:
            return []

        # Reorder columns to have file_name first
        cols = ["file_name", "model_name"] + [
            c for c in df.columns if c not in ["file_name", "model_name"]
        ]
        df = df[cols]

        # Sort the DataFrame
        df = df.sort_values(by=["file_name", "model_name"])

        # For visualization, we need to average the metrics per model
        # but the CSV will contain the detailed data.
        df_for_plots = df.drop(columns=["file_name"]).groupby("model_name").mean()

        if "discriminant_score" in df_for_plots.columns:
            df_for_plots = df_for_plots.sort_values(
                by="discriminant_score", ascending=False
            )

        # Export detailed metrics to CSV
        csv_path = self.output_dir / f"{file_prefix}_metrics_comparison.csv"
        df.to_csv(csv_path, index=False)
        logging.info(f"Exported metrics for '{file_prefix}' to {csv_path}")

        # Create a dataframe for the radar chart, which will be filtered by top_n
        df_for_radar = df_for_plots.copy()
        if top_n:
            df_for_radar = df_for_radar.head(top_n)

        metric_preferences = {
            "intra_cluster_distance_normalized": True,
            "inter_cluster_distance_normalized": True,
            "silhouette_score": True,
            "local_density_index": True,
            "internal_coherence_score": False,
            "robustness_score": True,
        }

        metrics_to_plot = [m for m in metric_preferences if m in df_for_plots.columns]
        if not metrics_to_plot:
            return []

        plot_paths = []
        for metric in metrics_to_plot:
            plot_path = self.output_dir / f"{file_prefix}_{metric}_comparison.png"
            # Pass the aggregated dataframe for plotting
            self._plot_single_metric(
                df_for_plots,
                metric,
                plot_path,
                higher_is_better=metric_preferences[metric],
                top_n=top_n,
            )
            plot_paths.append(plot_path)

        # Generate and add radar chart using the potentially filtered dataframe
        radar_path = self._generate_radar_chart(df_for_radar, file_prefix)
        if radar_path:
            plot_paths.append(radar_path)

        return plot_paths


def get_metrics_info():
    """Return information about metrics including names, descriptions, and whether higher is better."""
    return {
        "intra_cluster_distance_normalized": {
            "name": "Intra-Cluster Quality",
            "description": "Normalized intra-cluster distance (cohesion within themes)",
            "higher_is_better": True,
            "range": "0-1",
        },
        "inter_cluster_distance_normalized": {
            "name": "Inter-Cluster Separation",
            "description": "Normalized inter-cluster distance (separation between themes)",
            "higher_is_better": True,
            "range": "0-1",
        },
        "silhouette_score": {
            "name": "Silhouette Score",
            "description": "Overall clustering quality measure",
            "higher_is_better": True,
            "range": "-1 to 1",
        },
        "local_density_index": {
            "name": "Local Density Index",
            "description": "Proportion of neighbors sharing the same theme",
            "higher_is_better": True,
            "range": "0-1",
        },
        "internal_coherence_score": {
            "name": "Internal Coherence Score",
            "description": "Stability of similarity measurements",
            "higher_is_better": False,
            "range": "0+",
        },
        "robustness_score": {
            "name": "Robustness Score",
            "description": "Stability against noise and perturbations",
            "higher_is_better": True,
            "range": "0-1",
        },
    }
