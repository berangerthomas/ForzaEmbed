"""Report generation module for ForzaEmbed.

This module provides the ReportGenerator class that handles the generation
of all reports and visualizations, including comparison charts, radar charts,
and interactive web pages.

Example:
    Generate reports from processing results::

        from src.reporting.reporting import ReportGenerator

        generator = ReportGenerator(db, config, output_dir, "config_name")
        generator.generate_all(top_n=25, single_file=False)
"""

import logging
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..utils.database import EmbeddingDatabase
from .aggregator import DataAggregator
from .markdown_filter import MarkdownFilter


class ReportGenerator:
    """Handle the generation of all reports and visualizations.

    Coordinates the generation of comparison charts, radar charts, filtered
    markdowns, and interactive web pages from processing results.

    Attributes:
        db: The embedding database containing results.
        config: Configuration dictionary with report settings.
        output_dir: Directory path for output files.
        config_name: Name of the configuration for file prefixes.
        similarity_threshold: Threshold for similarity-based filtering.
        data_aggregator: Helper for aggregating data from database.
        markdown_filter: Helper for generating filtered markdowns.
    """

    def __init__(
        self,
        db: EmbeddingDatabase,
        config: dict[str, Any],
        output_dir: Path,
        config_name: str,
    ) -> None:
        """Initialize the ReportGenerator.

        Args:
            db: The embedding database containing results.
            config: Configuration dictionary with report settings.
            output_dir: Directory path for output files.
            config_name: Name of the configuration for file prefixes.
        """
        self.db = db
        self.config = config
        self.output_dir = output_dir
        self.config_name = config_name
        # Server-side similarity threshold removed; report generator no longer reads it.
        self.similarity_threshold = None
        self.data_aggregator = DataAggregator(db, output_dir, config_name)
        self.markdown_filter = MarkdownFilter(db, config, output_dir, config_name)

    def generate_all(
        self, top_n: int = 25, single_file: bool = False, data_source: str = "markdowns"
    ) -> None:
        """Generate all reports from the data in the database.

        Args:
            top_n: Maximum number of top models to include in reports.
                Use -1 for all models. Defaults to 25.
            single_file: If True, creates a single HTML file for all results.
                If False, creates one HTML per markdown file. Defaults to False.
            data_source: Source directory name for data files. Defaults to 'markdowns'.
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

        # Generate filtered markdowns
        self.markdown_filter.generate_filtered_markdowns()

        logging.info(f"All reports generated in '{self.output_dir}'.")
        self.data_aggregator.touch_cache()

    def _generate_main_web_page(
        self,
        processed_data: dict[str, Any],
        total_combinations: int,
        single_file: bool = False,
        graph_paths: dict[str, Any] | None = None,
    ) -> None:
        """Generate the main interactive web page.

        Args:
            processed_data: Processed data dictionary for visualization.
            total_combinations: Total number of model combinations processed.
            single_file: Whether to generate a single file or per-document files.
            graph_paths: Dictionary mapping file IDs to their graph paths.
        """
        from .web_generator import generate_main_page

        # Extract themes from config for display in tooltips
        themes_config: dict[str, Any] = {}
        if hasattr(self.config, 'grid_search_params'):
            themes_config = self.config.grid_search_params.themes
        elif isinstance(self.config, dict) and 'grid_search_params' in self.config:
            themes_config = self.config['grid_search_params'].get('themes', {})

        generate_main_page(
            processed_data,
            str(self.output_dir),
            total_combinations,
            single_file=single_file,
            graph_paths=graph_paths,
            config_name=self.config_name,
            themes_config=themes_config,
        )

    def _generate_global_reports(
        self,
        all_models_metrics: dict[str, list[dict[str, Any]]],
        top_n: int | None = None,
        file_prefix: str = "global",
    ) -> list[Path]:
        """Generate global comparison charts.

        Args:
            all_models_metrics: Dictionary mapping model names to their metrics.
            top_n: Maximum number of top models to include. None for all.
            file_prefix: Prefix for output files. Defaults to 'global'.

        Returns:
            List of paths to generated plot files.
        """
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
        """Generate and save a sorted bar plot for a single metric.

        Args:
            df: DataFrame with model names as index and metrics as columns.
            metric: Name of the metric column to plot.
            output_path: Path where to save the plot image.
            higher_is_better: If True, sorts descending; if False, ascending.
            top_n: Maximum number of models to show. None for all.
        """
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
        """Generate a radar chart for the most important metrics.

        Creates a polar plot comparing normalized metric values across
        models for key clustering quality metrics.

        Args:
            df: DataFrame with model names as index and metrics as columns.
            file_prefix: Prefix for the output file. Defaults to 'global'.

        Returns:
            Path to the generated radar chart image, or None if not enough metrics.
        """
        metrics_for_radar = {
            "silhouette_score": True,
            "inter_cluster_distance_normalized": True,
            "intra_cluster_distance_normalized": True,
            "embedding_computation_time": False,
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

        radar_path = (
            self.output_dir / f"{self.config_name}_{file_prefix}_radar_chart.png"
        )
        plt.savefig(radar_path, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved radar chart to {radar_path}")
        return radar_path

    def _analyze_and_visualize_clustering_metrics(
        self,
        all_models_metrics: dict[str, list[dict[str, Any]]],
        top_n: int | None = None,
        file_prefix: str = "global",
    ) -> list[Path]:
        """Analyze clustering metrics and generate visualization plots.

        Creates individual bar plots for each metric and a summary radar chart.
        Exports detailed metrics to CSV.

        Args:
            all_models_metrics: Dictionary mapping model names to lists of
                metric dictionaries.
            top_n: Maximum number of top models to include. None for all.
            file_prefix: Prefix for output files. Defaults to 'global'.

        Returns:
            List of paths to the generated plot files.
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

        if "silhouette_score" in df_for_plots.columns:
            df_for_plots = df_for_plots.sort_values(
                by="silhouette_score", ascending=False
            )

        # Export detailed metrics to CSV with config prefix
        csv_path = (
            self.output_dir / f"{self.config_name}_{file_prefix}_metrics_comparison.csv"
        )
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

        }

        metrics_to_plot = [m for m in metric_preferences if m in df_for_plots.columns]
        if not metrics_to_plot:
            return []

        plot_paths = []
        for metric in metrics_to_plot:
            plot_path = (
                self.output_dir
                / f"{self.config_name}_{file_prefix}_{metric}_comparison.png"
            )
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


def get_metrics_info() -> dict[str, dict[str, Any]]:
    """Return information about metrics including names, descriptions, and preferences.

    Returns:
        Dictionary mapping metric keys to their metadata:
            - name: Human-readable metric name.
            - description: Explanation of what the metric measures.
            - higher_is_better: Whether higher values indicate better performance.
            - range: Expected value range as a string.
    """
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
        "embedding_computation_time": {
            "name": "Embedding Computation Time",
            "description": "Time taken to compute embeddings (seconds)",
            "higher_is_better": False,
            "range": "0+",
        },
    }
