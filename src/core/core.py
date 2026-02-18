"""Core orchestration module for ForzaEmbed.

This module contains the main ForzaEmbed class that orchestrates the entire
embedding analysis pipeline, including grid search execution and report
generation. It manages the workflow for testing multiple parameter combinations
and generating comprehensive reports.

Example:
    Run a complete analysis::

        from src.core.core import ForzaEmbed

        app = ForzaEmbed(db_path="reports/results.db", config_path="configs/config.yml")
        app.run_grid_search(data_source="markdowns", resume=True)
        app.generate_reports(top_n=25)
"""

import itertools
import logging
from pathlib import Path
from typing import Any, Union

from tqdm import tqdm

from ..reporting.reporting import ReportGenerator
from ..utils.data_loader import load_markdown_files
from ..utils.database import EmbeddingDatabase
from .config import AppConfig, ModelConfig, load_config
from .processing import Processor


# Chunking strategies that ignore chunk_size and chunk_overlap parameters
# Based on their implementation:
# - nltk: Uses sentence tokenization (nltk.sent_tokenize) - ignores size/overlap
# - spacy: Uses spaCy's sentence segmentation - ignores size/overlap
PARAMETER_INSENSITIVE_STRATEGIES: set[str] = {"nltk", "spacy"}

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ForzaEmbed:
    """Main orchestrator for the embedding analysis and reporting pipeline.

    This class manages the complete workflow for embedding analysis, including
    loading configurations, running grid searches across multiple parameter
    combinations, and generating comprehensive reports.

    Attributes:
        db_path: Path to the SQLite database file.
        config_path: Path to the YAML configuration file.
        config: The loaded application configuration.
        config_name: Name derived from the configuration file.
        db: The embedding database instance.
        output_dir: Directory for output files.
        processor: The data processor instance.
        report_generator: The report generator instance.
    """

    def __init__(
        self,
        db_path: str = "reports/config_ForzaEmbed.db",
        config_path: str = "configs/config.yml",
    ) -> None:
        """Initialize the ForzaEmbed instance.

        Args:
            db_path: Path to the SQLite database file for storing results.
            config_path: Path to the YAML configuration file.
        """
        self.db_path = Path(db_path)
        self.config_path = Path(config_path)
        self.config: AppConfig = load_config(config_path)

        # Extract config name for prefixing files
        self.config_name = self.config_path.stem

        # Initialize database with config parameter only
        self.db = EmbeddingDatabase(str(self.db_path), self.config.model_dump())

        # Ensure output directory exists - always use reports directory
        self.output_dir = Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Instantiate the processor
        self.processor = Processor(self.db, self.config)
        self.report_generator = ReportGenerator(
            self.db, self.config.model_dump(), self.output_dir, self.config_name
        )

        logging.info(f"ForzaEmbed initialized. Database at: {self.db_path}")
        logging.info(f"Output directory: {self.output_dir}")
        logging.info(f"Config prefix: {self.config_name}")

    def _generate_smart_combinations(self, param_grid: dict[str, Any]) -> list[tuple[Any, ...]]:
        """Generate parameter combinations intelligently.

        Avoids redundant combinations for chunking strategies that don't use
        chunk_size/chunk_overlap parameters (like nltk and spacy).

        Args:
            param_grid: Dictionary mapping parameter names to lists of values.

        Returns:
            List of valid parameter combination tuples.
        """
        strategies = param_grid["chunking_strategy"]
        chunk_sizes = param_grid["chunk_size"]
        chunk_overlaps = param_grid["chunk_overlap"]

        # Separate strategies that are parameter-sensitive and parameter-insensitive
        sensitive_strategies = [s for s in strategies if s not in PARAMETER_INSENSITIVE_STRATEGIES]
        insensitive_strategies = [s for s in strategies if s in PARAMETER_INSENSITIVE_STRATEGIES]

        combinations = []

        # For parameter-sensitive strategies (langchain, semchunk, raw),
        # generate all combinations of chunk_size and chunk_overlap
        if sensitive_strategies:
            sensitive_params = {
                "model_config": param_grid["model_config"],
                "chunk_size": chunk_sizes,
                "chunk_overlap": chunk_overlaps,
                "chunking_strategy": sensitive_strategies,
                "similarity_metrics": param_grid["similarity_metrics"],
                "theme_name": param_grid["theme_name"],
            }
            sensitive_combinations = list(itertools.product(*sensitive_params.values()))
            # Filter: chunk_size > chunk_overlap
            sensitive_combinations = [
                params for params in sensitive_combinations
                if params[1] > params[2]  # chunk_size > chunk_overlap
            ]
            combinations.extend(sensitive_combinations)

        # For parameter-insensitive strategies (nltk, spacy),
        # use only one chunk_size and one chunk_overlap (the first values)
        # since these parameters are ignored anyway
        if insensitive_strategies:
            # Use the smallest chunk_size and chunk_overlap (first values) as dummy values
            dummy_chunk_size = chunk_sizes[0] if chunk_sizes else 100
            dummy_chunk_overlap = chunk_overlaps[0] if chunk_overlaps else 0

            insensitive_params = {
                "model_config": param_grid["model_config"],
                "chunk_size": [dummy_chunk_size],
                "chunk_overlap": [dummy_chunk_overlap],
                "chunking_strategy": insensitive_strategies,
                "similarity_metrics": param_grid["similarity_metrics"],
                "theme_name": param_grid["theme_name"],
            }
            insensitive_combinations = list(itertools.product(*insensitive_params.values()))
            combinations.extend(insensitive_combinations)

        logging.info(
            f"Smart combination generation: "
            f"{len(sensitive_strategies)} parameter-sensitive strategies "
            f"({', '.join(sensitive_strategies) if sensitive_strategies else 'none'}), "
            f"{len(insensitive_strategies)} parameter-insensitive strategies "
            f"({', '.join(insensitive_strategies) if insensitive_strategies else 'none'})"
        )

        return combinations

    def run_grid_search(
        self, data_source: Union[str, Path, list[str]], resume: bool = True
    ) -> None:
        """Run the complete grid search pipeline.

        Executes the embedding analysis across all parameter combinations
        defined in the configuration. Supports resumption from the last
        completed combination.

        Args:
            data_source: The source of markdown data. Can be a directory path
                (str or Path) or a list of markdown content strings.
            resume: If True, resumes from the last completed combination.
        """
        self.data_source = data_source  # Store data_source
        logging.info("--- Starting Data Processing ---")

        all_rows = load_markdown_files(data_source)
        logging.info(
            f"Found {len(all_rows)} markdown files to process from '{data_source}'"
        )
        all_rows.sort(key=lambda x: x[0])

        param_grid = {
            "model_config": self.config.models_to_test,
            "chunk_size": self.config.grid_search_params.chunk_size,
            "chunk_overlap": self.config.grid_search_params.chunk_overlap,
            "chunking_strategy": self.config.grid_search_params.chunking_strategy,
            "similarity_metrics": self.config.grid_search_params.similarity_metrics,
            "theme_name": list(self.config.grid_search_params.themes.keys()),
        }

        # Generate smart combinations that avoid redundant calculations
        # for chunking strategies that don't use chunk_size/chunk_overlap
        valid_combinations = self._generate_smart_combinations(param_grid)

        # Calculate the exact number of tasks to be processed for an accurate progress bar
        logging.info("Calculating exact number of tasks to process...")
        total_tasks = 0
        for params in valid_combinations:
            run_name = self._generate_run_name(*params)
            processed_files = self.db.get_processed_files(run_name)
            unprocessed_count = len(
                [row for row in all_rows if row[0] not in processed_files]
            )
            total_tasks += unprocessed_count

        num_files = len(all_rows)
        num_combinations = len(valid_combinations)
        total_possible_tasks = num_files * num_combinations
        cached_tasks = total_possible_tasks - total_tasks

        file_str = f"{num_files} file{'s' if num_files > 1 else ''}"
        combination_str = (
            f"{num_combinations} valid combination{'s' if num_combinations > 1 else ''}"
        )

        log_message = (
            f"Found {file_str} to process with {combination_str}. "
            f"This represents a total of {total_possible_tasks} possible calculations. "
        )

        if cached_tasks > 0:
            log_message += (
                f"Found {cached_tasks} already completed calculations. "
                f"Resuming processing for the remaining {total_tasks} calculations."
            )
        else:
            log_message += f"Starting processing for all {total_tasks} calculations."

        logging.info(log_message)

        if total_tasks == 0:
            logging.info("All combinations already processed for all files!")
            return

        with tqdm(total=total_tasks, desc="Processing files") as pbar:
            for params in valid_combinations:
                (
                    model_config,
                    chunk_size,
                    chunk_overlap,
                    chunking_strategy,
                    similarity_metric,
                    theme_name,
                ) = params
                run_name = self._generate_run_name(*params)

                processed_files = self.db.get_processed_files(run_name)
                themes = self.config.grid_search_params.themes[theme_name]

                result = self.processor.run_test(
                    rows=all_rows,
                    model_config=model_config,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    themes=themes,
                    theme_name=theme_name,
                    chunking_strategy=chunking_strategy,
                    similarity_metric=similarity_metric,
                    processed_files=processed_files,
                    pbar=pbar,
                )

                model_results = result.get("results", {})
                results_to_save = [
                    (run_name, file_id, file_data)
                    for file_id, file_data in model_results.get("files", {}).items()
                    if file_data
                ]
                if results_to_save:
                    self.db.add_model(
                        run_name,
                        model_config.name,
                        model_config.type,
                        chunk_size,
                        chunk_overlap,
                        theme_name,
                        chunking_strategy,
                        similarity_metric,
                    )
                    self.db.save_processing_results_batch(results_to_save)
        logging.info("--- Grid Search Finished ---")

    def _generate_run_name(
        self,
        model_config: ModelConfig,
        chunk_size: int,
        chunk_overlap: int,
        chunking_strategy: str,
        similarity_metric: str,
        theme_name: str,
    ) -> str:
        """Generate a standardized run name for a parameter combination.

        Args:
            model_config: The model configuration.
            chunk_size: The chunk size parameter.
            chunk_overlap: The chunk overlap parameter.
            chunking_strategy: The chunking strategy name.
            similarity_metric: The similarity metric name.
            theme_name: The theme set name.

        Returns:
            A unique string identifier for the parameter combination.
        """
        model_name = model_config.name.replace("/", "_")
        dimensions = model_config.dimensions
        return f"{model_name}_d{dimensions}_cs{chunk_size}_co{chunk_overlap}_t{theme_name}_s{chunking_strategy}_m{similarity_metric}"

    def generate_reports(self, top_n: int = 25, single_file: bool = False) -> None:
        """Generate all reports and visualizations.

        Creates comprehensive reports from the data stored in the database,
        including metric comparisons, charts, and interactive visualizations.

        Args:
            top_n: Number of top combinations to include in reports.
                Use -1 to include all combinations.
            single_file: If True, generates a single HTML file containing
                all results. If False, generates separate files per input.
        """
        # Use stored data_source or default to "markdowns"
        data_source = getattr(self, 'data_source', 'markdowns')
        self.report_generator.generate_all(
            top_n=top_n, single_file=single_file, data_source=data_source
        )


if __name__ == "__main__":
    # Example usage:
    # This allows for testing the class directly
    app = ForzaEmbed()
    # Note: Provide a default data source for direct script execution
    app.run_grid_search(data_source="data/markdown")
    app.generate_reports(top_n=-1)
