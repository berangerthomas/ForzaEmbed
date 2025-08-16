import itertools
import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..reporting.reporting import ReportGenerator
from ..utils.data_loader import load_markdown_files
from ..utils.database import EmbeddingDatabase
from .config import AppConfig, load_config
from .processing import Processor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ForzaEmbed:
    """
    Main class for managing the embedding grid search and reporting pipeline.
    """

    def __init__(
        self,
        db_path: str = "data/ForzaEmbed.db",
        config_path: str = "configs/config.yml",
    ):
        """
        Initializes the ForzaEmbed instance.

        Args:
            db_path (str): Path to the SQLite database file.
            config_path (str): Path to the YAML configuration file.
        """
        self.db_path = Path(db_path)
        self.config_path = Path(config_path)
        self.config: AppConfig = load_config(config_path)

        # Initialize database with config parameter only
        self.db = EmbeddingDatabase(str(self.db_path), self.config.model_dump())

        # Ensure output directory exists
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Instantiate the processor
        self.processor = Processor(self.db, self.config)
        self.report_generator = ReportGenerator(
            self.db, self.config.model_dump(), self.output_dir
        )

        logging.info(f"ForzaEmbed initialized. Database at: {self.db_path}")
        logging.info(f"Output directory: {self.output_dir}")

    def run_grid_search(self, data_source: Any, resume: bool = True):
        """
        Runs the entire grid search pipeline.

        Args:
            data_source (Any): The source of the markdown data. Can be a
                               directory path (str or Path) or a list of
                               markdown content strings.
            resume (bool): If True, resumes from the last completed combination.
        """
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

        param_combinations = list(itertools.product(*param_grid.values()))
        valid_combinations = [
            params
            for params in param_combinations
            if params[1] > params[2]  # chunk_size > chunk_overlap
        ]

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

        logging.info(
            f"Generated {len(valid_combinations)} valid combinations. "
            f"Found {total_tasks} file(s) to process across all combinations."
        )

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
        model_config,
        chunk_size,
        chunk_overlap,
        chunking_strategy,
        similarity_metric,
        theme_name,
    ):
        """Generates a standardized run name for a parameter combination."""
        model_name = model_config.name.replace("/", "_")
        dimensions = model_config.dimensions
        return f"{model_name}_d{dimensions}_cs{chunk_size}_co{chunk_overlap}_t{theme_name}_s{chunking_strategy}_m{similarity_metric}"

    def generate_reports(self, top_n: int = 25, single_file: bool = False):
        """
        Generates all reports and visualizations.

        Args:
            top_n (int): Number of top combinations to include in reports.
                         Use -1 for all.
            single_file (bool): If True, generates a single HTML file.
        """
        self.report_generator.generate_all(top_n=top_n, single_file=single_file)


if __name__ == "__main__":
    # Example usage:
    # This allows for testing the class directly
    app = ForzaEmbed()
    # Note: Provide a default data source for direct script execution
    app.run_grid_search(data_source="data/markdown")
    app.generate_reports(top_n=-1)
