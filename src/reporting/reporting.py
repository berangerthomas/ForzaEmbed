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
            graph_paths_by_file["global"] = {}
        else:
            for file_id in processed_data_for_interactive_page["files"]:
                graph_paths_by_file[file_id] = {}

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
