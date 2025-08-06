import itertools
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from tqdm import tqdm

from .data_loader import load_markdown_files
from .database import EmbeddingDatabase
from .processing import Processor
from .reporting import ReportGenerator

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
        grid_search_params: Optional[Dict[str, Any]] = None,
        models_to_test: Optional[List[Dict[str, Any]]] = None,
        config_path: str = "config.yml",
    ):
        """
        Initializes the ForzaEmbed instance.

        Configuration can be provided programmatically via `grid_search_params`
        and `models_to_test`. If they are not provided, it falls back to
        loading the configuration from the `config_path` YAML file.

        Args:
            db_path (str): Path to the SQLite database file.
            grid_search_params (Optional[Dict]): Dictionary of grid search parameters.
            models_to_test (Optional[List]): List of models to test.
            config_path (str): Path to the YAML configuration file.
        """
        self.db_path = Path(db_path)
        self.db = EmbeddingDatabase(str(self.db_path))
        self.config = self._load_config(grid_search_params, models_to_test, config_path)

        # Ensure output directory exists
        self.output_dir = Path(self.config.get("output_dir", "data/output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Instantiate the processor
        self.processor = Processor(self.db, self.config)
        self.report_generator = ReportGenerator(self.db, self.config, self.output_dir)

        logging.info(f"ForzaEmbed initialized. Database at: {self.db_path}")
        logging.info(f"Output directory: {self.output_dir}")

    def _load_config(
        self,
        grid_search_params: Optional[Dict[str, Any]],
        models_to_test: Optional[List[Dict[str, Any]]],
        config_path: str,
    ) -> Dict[str, Any]:
        """Loads configuration from arguments or a YAML file."""
        if grid_search_params and models_to_test:
            logging.info("Loading configuration from programmatic arguments.")
            return {
                "grid_search_params": grid_search_params,
                "models_to_test": models_to_test,
            }
        else:
            logging.info(f"Loading configuration from YAML file: {config_path}")
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            except FileNotFoundError:
                logging.error(f"Configuration file not found at: {config_path}")
                raise
            except yaml.YAMLError as e:
                logging.error(f"Error parsing YAML file: {e}")
                raise

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
            "model_config": self.config["models_to_test"],
            **self.config["grid_search_params"],
        }
        # Replace theme names with the actual theme lists
        param_grid["theme_name"] = list(param_grid["themes"].keys())
        del param_grid["themes"]  # We only need the names for combinations

        param_combinations = list(itertools.product(*param_grid.values()))
        valid_combinations = [
            params
            for params in param_combinations
            if params[1] > params[2]  # chunk_size > chunk_overlap
        ]

        logging.info("Checking for already completed combinations...")
        completed_combinations = set()
        if resume:
            completed_combinations = self._get_completed_combinations(all_rows)

        remaining_combinations = [
            params
            for params in valid_combinations
            if self._generate_run_name(*params) not in completed_combinations
        ]

        completed_count = len(valid_combinations) - len(remaining_combinations)
        total_tasks = len(remaining_combinations) * len(all_rows)
        logging.info(
            f"Generated {len(valid_combinations)} valid combinations. "
            f"Found {completed_count} completed. "
            f"Remaining: {len(remaining_combinations)} combinations, for a total of {total_tasks} calculations."
        )

        if not remaining_combinations:
            logging.info("All combinations already processed!")
            return

        with tqdm(total=total_tasks, desc="Processing files") as pbar:
            for params in remaining_combinations:
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
                themes = self.config["grid_search_params"]["themes"][theme_name]

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
                        model_config["name"],
                        model_config["type"],
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
        model_name = model_config["name"]
        dimensions = model_config.get("dimensions", "auto")
        return f"{model_name}_d{dimensions}_cs{chunk_size}_co{chunk_overlap}_t{theme_name}_s{chunking_strategy}_m{similarity_metric}"

    def _get_completed_combinations(self, all_rows: list) -> set[str]:
        """
        Retrieves the list of fully processed combinations from the database.
        """
        completed_combinations = set()
        all_run_names = self.db.get_all_run_names()
        total_files = len(all_rows)

        for run_name in all_run_names:
            processed_files = self.db.get_processed_files_with_similarities(run_name)
            if len(processed_files) == total_files:
                completed_combinations.add(run_name)
        return completed_combinations

    def generate_reports(
        self, all_combinations: bool = False, single_file: bool = False
    ):
        """
        Generates all reports and visualizations.

        Args:
            all_combinations (bool): If True, uses all combinations for charts.
            single_file (bool): If True, generates a single HTML file.
        """
        top_n = None if all_combinations else 25
        self.report_generator.generate_all(top_n=top_n, single_file=single_file)

    def _setup_database(self):
        """Set up the database schema."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            model_name TEXT NOT NULL,
            chunk_method TEXT NOT NULL,
            chunk_size INTEGER,
            chunk_overlap INTEGER,
            similarity_metric TEXT NOT NULL,
            similarity_threshold REAL NOT NULL,
            min_chunks_per_theme INTEGER NOT NULL,
            max_themes INTEGER NOT NULL,
            num_chunks INTEGER,
            num_themes INTEGER,
            intra_cluster_distance_normalized REAL,
            inter_cluster_distance_normalized REAL,
            silhouette_score REAL,
            local_density_index REAL,
            internal_coherence_score REAL,
            robustness_score REAL,
            execution_time REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(filename, model_name, chunk_method, chunk_size, chunk_overlap, similarity_metric, similarity_threshold, min_chunks_per_theme, max_themes)
        )
        """
        self.cursor.execute(create_table_sql)
        self.db_connection.commit()

    def _save_results(self, filename: str, config: dict, results: dict):
        """Save results to the database."""
        insert_sql = """
        INSERT OR REPLACE INTO results (
            filename, model_name, chunk_method, chunk_size, chunk_overlap,
            similarity_metric, similarity_threshold, min_chunks_per_theme, max_themes,
            num_chunks, num_themes, intra_cluster_distance_normalized, inter_cluster_distance_normalized,
            silhouette_score, local_density_index, internal_coherence_score, robustness_score,
            execution_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        values = (
            filename,
            config["model_name"],
            config["chunk_method"],
            config.get("chunk_size"),
            config.get("chunk_overlap"),
            config["similarity_metric"],
            config["similarity_threshold"],
            config["min_chunks_per_theme"],
            config["max_themes"],
            results.get("num_chunks"),
            results.get("num_themes"),
            results.get("intra_cluster_distance_normalized"),
            results.get("inter_cluster_distance_normalized"),
            results.get("silhouette_score"),
            results.get("local_density_index"),
            results.get("internal_coherence_score"),
            results.get("robustness_score"),
            results.get("execution_time"),
        )

        self.cursor.execute(insert_sql, values)
        self.db_connection.commit()

    def refresh_metrics(self):
        """Refresh evaluation metrics for all existing runs."""
        logging.info("Refreshing evaluation metrics for all existing runs...")

        # Get all unique combinations that have been run
        query = """
        SELECT DISTINCT filename, model_name, chunk_method, chunk_size, chunk_overlap,
                       similarity_metric, similarity_threshold, min_chunks_per_theme, max_themes
        FROM results
        """
        self.cursor.execute(query)
        combinations = self.cursor.fetchall()

        for combination in combinations:
            (
                filename,
                model_name,
                chunk_method,
                chunk_size,
                chunk_overlap,
                similarity_metric,
                similarity_threshold,
                min_chunks_per_theme,
                max_themes,
            ) = combination

            config = {
                "model_name": model_name,
                "chunk_method": chunk_method,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "similarity_metric": similarity_metric,
                "similarity_threshold": similarity_threshold,
                "min_chunks_per_theme": min_chunks_per_theme,
                "max_themes": max_themes,
            }

            try:
                # Re-run the processing to get fresh metrics
                results = self._process_single_combination(filename, config)

                # Update only the metrics columns
                update_sql = """
                UPDATE results SET
                    intra_cluster_distance_normalized = ?,
                    inter_cluster_distance_normalized = ?,
                    silhouette_score = ?,
                    local_density_index = ?,
                    internal_coherence_score = ?,
                    robustness_score = ?
                WHERE filename = ? AND model_name = ? AND chunk_method = ? AND
                      chunk_size = ? AND chunk_overlap = ? AND similarity_metric = ? AND
                      similarity_threshold = ? AND min_chunks_per_theme = ? AND max_themes = ?
                """

                update_values = (
                    results.get("intra_cluster_distance_normalized"),
                    results.get("inter_cluster_distance_normalized"),
                    results.get("silhouette_score"),
                    results.get("local_density_index"),
                    results.get("internal_coherence_score"),
                    results.get("robustness_score"),
                    filename,
                    model_name,
                    chunk_method,
                    chunk_size,
                    chunk_overlap,
                    similarity_metric,
                    similarity_threshold,
                    min_chunks_per_theme,
                    max_themes,
                )

                self.cursor.execute(update_sql, update_values)
                self.db_connection.commit()

                logging.info(f"Refreshed metrics for {filename} with {model_name}")

            except Exception as e:
                logging.error(
                    f"Error refreshing metrics for {filename} with {model_name}: {e}"
                )

        logging.info("Finished refreshing evaluation metrics.")

    def clear_database(self):
        """Clears all data from the main database."""
        logging.warning("Clearing the main database...")
        self.db.clear_database()
        logging.info("Main database cleared.")

    def clear_embedding_cache(self):
        """Clears the embedding cache."""
        logging.warning("Clearing the embedding cache...")
        self.db.clear_embedding_cache()
        logging.info("Embedding cache cleared.")


if __name__ == "__main__":
    # Example usage:
    # This allows for testing the class directly
    app = ForzaEmbed()
    # Note: Provide a default data source for direct script execution
    app.run_grid_search(data_source="data/markdown")
    app.generate_reports(all_combinations=True)
