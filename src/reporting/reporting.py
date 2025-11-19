import logging
import textwrap
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..utils.database import EmbeddingDatabase
from .aggregator import DataAggregator
from .markdown_filter import MarkdownFilter


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
        self.config_name = config_name
        self.similarity_threshold = config.get("similarity_threshold", 0.6)
        self.data_aggregator = DataAggregator(db, output_dir, config_name)
        self.markdown_filter = MarkdownFilter(db, config, output_dir, config_name)

    def generate_all(
        self, top_n: int = 25, single_file: bool = False, data_source: str = "markdowns"
    ):
        """
        Generates all reports from the data in the database.
        """
        logging.info("--- Generating All Reports ---")
        effective_top_n = None if top_n == -1 else top_n
        aggregated_data = self.data_aggregator.get_aggregated_data()

        if not aggregated_data:
            logging.warning("No aggregated data available. Skipping report generation.")
            return

        df = self._generate_filtered_markdowns_report(aggregated_data, data_source)

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
            size_reduction_path = self._plot_size_reduction(df, file_prefix="global")
            if size_reduction_path:
                global_plot_paths.append(size_reduction_path)
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
                    size_reduction_path = None
                    if not df.empty and "original_file" in df.columns:
                        size_reduction_path = self._plot_size_reduction(
                            df[df["original_file"] == file_id], file_prefix=file_prefix
                        )
                    if size_reduction_path:
                        plot_paths.append(size_reduction_path)
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

    def _reconstruct_content_without_overlap(self, phrases: list[str]) -> str:
        """
        Reconstruit le contenu à partir d'une liste de phrases (chunks)
        en éliminant les chevauchements.

        Args:
            phrases: Liste des phrases filtrées.

        Returns:
            Texte reconstitué sans duplication.
        """
        if not phrases:
            return ""

        # Trier les phrases par ordre d'apparition implicite (en supposant que l'ordre initial est conservé)
        # et nettoyer les espaces superflus.
        unique_phrases = sorted(list(set(phrases)), key=phrases.index)

        reconstructed_text = unique_phrases[0]

        for i in range(1, len(unique_phrases)):
            next_phrase = unique_phrases[i]

            # Trouver la plus grande longueur de chevauchement possible
            overlap_len = 0
            for j in range(min(len(reconstructed_text), len(next_phrase)), 0, -1):
                if reconstructed_text.endswith(next_phrase[:j]):
                    overlap_len = j
                    break

            # Ajouter uniquement la partie non chevauchante de la phrase suivante
            reconstructed_text += next_phrase[overlap_len:]

        return reconstructed_text

    def _generate_filtered_markdowns_report(
        self, aggregated_data: Dict[str, Any], data_source: str
    ):
        """
        Generates filtered markdown files and a CSV report comparing their sizes.
        """
        if not self.config.get("generate_filtered_markdowns"):
            logging.info(
                "Skipping filtered markdown generation as 'generate_filtered_markdowns' is false."
            )
            return pd.DataFrame()

        logging.info("--- Generating Filtered Markdowns and Size Comparison Report ---")

        # Filtered markdowns should be at the root of the reports directory
        filtered_md_dir = self.output_dir

        size_comparison_data = []

        processed_data = aggregated_data.get("processed_data_for_interactive_page", {})
        all_files_data = processed_data.get("files", {})

        if not all_files_data:
            logging.warning(
                "No file data found in aggregated data. Skipping markdown generation."
            )
            return pd.DataFrame()

        for file_id, file_data in all_files_data.items():
            original_file_path = Path(data_source) / f"{file_id}.md"
            try:
                original_content = original_file_path.read_text(encoding="utf-8")
                original_size = len(original_content)
            except FileNotFoundError:
                logging.warning(
                    f"Original markdown file not found: {file_id}. Skipping."
                )
                continue
            except Exception as e:
                logging.error(f"Error reading file {file_id}: {e}. Skipping.")
                continue

            # Add original file data to the report
            size_comparison_data.append(
                {
                    "original_file": file_id,
                    "prefix": "original",
                    "filtered_size_chars": original_size,
                    "percentage_of_original": 100.0,
                }
            )

            for model_name, embedding_data in file_data.get("embeddings", {}).items():
                phrases = embedding_data.get("phrases", [])
                similarities = embedding_data.get("similarities", [])

                if not phrases or not similarities:
                    continue

                filtered_phrases = [
                    phrase
                    for phrase, sim in zip(phrases, similarities)
                    if sim >= self.similarity_threshold
                ]

                # Simplement concaténer sans ajouter de séparateur comme demandé
                if filtered_phrases:
                    filtered_content = self._reconstruct_content_without_overlap(
                        filtered_phrases
                    )
                else:
                    filtered_content = ""
                filtered_size = len(filtered_content)

                # Sanitize model_name for file path
                sanitized_model_name = model_name.replace("/", "_").replace("\\", "_")
                output_md_path = (
                    filtered_md_dir / f"{Path(file_id).stem}_{sanitized_model_name}.md"
                )

                try:
                    output_md_path.write_text(filtered_content, encoding="utf-8")
                except Exception as e:
                    logging.error(
                        f"Error writing filtered markdown {output_md_path}: {e}"
                    )
                    continue

                percentage = (
                    (filtered_size / original_size) * 100 if original_size > 0 else 0
                )

                size_comparison_data.append(
                    {
                        "original_file": file_id,
                        "prefix": model_name,
                        "filtered_size_chars": filtered_size,
                        "percentage_of_original": round(percentage, 2),
                    }
                )

        if size_comparison_data:
            df = pd.DataFrame(size_comparison_data)
            # Sort by filtered size
            df = df.sort_values(by="filtered_size_chars", ascending=True)

            csv_path = (
                self.output_dir
                / f"{self.config_name}_filtered_markdowns_size_comparison.csv"
            )
            df.to_csv(csv_path, index=False)
            logging.info(
                f"Generated filtered markdowns size comparison report: {csv_path}"
            )
        else:
            df = pd.DataFrame()
            logging.info("No data to generate a size comparison report.")

        return df

    def _plot_size_reduction(
        self, df: pd.DataFrame, top_n: int = 25, file_prefix: str = "global"
    ):
        """Generates and saves a bar plot for size reduction."""
        if df.empty or "prefix" not in df.columns:
            logging.info("No data available for size reduction plot.")
            return None

        # Exclude 'original' entries and average by prefix
        df_filtered = df[df["prefix"] != "original"]
        if df_filtered.empty:
            logging.info("No filtered data to generate size reduction plot.")
            return None

        df_agg = (
            df_filtered.groupby("prefix")["percentage_of_original"].mean().reset_index()
        )

        df_plot = df_agg.sort_values(by="percentage_of_original", ascending=True).head(
            top_n
        )

        plt.figure(figsize=(18, 12))
        ax = sns.barplot(
            x="prefix",
            y="percentage_of_original",
            data=df_plot,
            palette="viridis",
            hue="prefix",
            legend=False,
        )

        ax.set_title(
            "Top Model Configurations by Average Size Reduction",
            pad=20,
            fontsize=18,
        )
        ax.set_ylabel("Average Percentage of Original Size", fontsize=14)
        ax.set_xlabel("Model Configuration", fontsize=14)

        labels = [
            textwrap.fill(label, width=30, break_long_words=False)
            for label in df_plot["prefix"]
        ]
        ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")

        plt.tight_layout(pad=3.0)
        output_path = (
            self.output_dir
            / f"{self.config_name}_{file_prefix}_size_reduction_comparison.png"
        )
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Saved size reduction plot to {output_path}")
        return output_path

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
            config_name=self.config_name,
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
        "embedding_computation_time": {
            "name": "Embedding Computation Time",
            "description": "Time taken to compute embeddings (seconds)",
            "higher_is_better": False,
            "range": "0+",
        },
    }
