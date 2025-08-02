import argparse
import itertools
import multiprocessing
import os
import time

import numpy as np
from tqdm import tqdm

from src.config import (
    CMAP,
    GRID_SEARCH_PARAMS,
    MODELS_TO_TEST,
    OUTPUT_DIR,
    SIMILARITY_METRICS,
    SIMILARITY_THRESHOLD,
)
from src.data_loader import load_markdown_files
from src.database import EmbeddingDatabase
from src.lancedb_client import LanceDBClient
from src.processing import get_text_hash, run_test
from src.reporting import (
    analyze_and_visualize_clustering_metrics,
    analyze_and_visualize_variance,
    generate_explanatory_markdown,
    generate_filtered_markdown,
    generate_heatmap_html,
)
from src.utils import chunk_text
from src.web_generator import generate_main_page


def get_completed_combinations(db: EmbeddingDatabase, all_rows: list) -> set[str]:
    """
    Retrieves the list of fully processed combinations from the database.
    """
    completed_combinations = set()
    all_run_names = db.get_all_run_names()
    total_files = len(all_rows)

    for run_name in all_run_names:
        processed_files_with_similarities = db.get_processed_files_with_similarities(
            run_name
        )
        if len(processed_files_with_similarities) == total_files:
            completed_combinations.add(run_name)
    return completed_combinations


def generate_run_name(
    model_config,
    chunk_size,
    chunk_overlap,
    theme_name,
    chunking_strategy,
    similarity_metric,
):
    """Generates a standardized run name for a parameter combination."""
    model_name = model_config["name"]
    dimensions = model_config.get("dimensions", "auto")
    return f"{model_name}_d{dimensions}_cs{chunk_size}_co{chunk_overlap}_t{theme_name}_s{chunking_strategy}_m{similarity_metric}"


def run_processing(db: EmbeddingDatabase, lance_db: LanceDBClient):
    """Processes data using an optimized architecture with SQLite and LanceDB."""
    print("--- Starting Data Processing ---")

    markdown_directory = os.path.join(os.path.dirname(__file__), "data", "markdown")
    all_rows = load_markdown_files(markdown_directory)
    all_rows.sort(key=lambda x: x[0])

    param_grid = {
        "model_config": MODELS_TO_TEST,
        "chunk_size": GRID_SEARCH_PARAMS["chunk_size"],
        "chunk_overlap": GRID_SEARCH_PARAMS["chunk_overlap"],
        "theme_name": list(GRID_SEARCH_PARAMS["themes"].keys()),
        "chunking_strategy": GRID_SEARCH_PARAMS["chunking_strategy"],
        "similarity_metric": SIMILARITY_METRICS,
    }

    param_combinations = list(itertools.product(*param_grid.values()))
    valid_combinations = [
        params for params in param_combinations if params[1] > params[2]
    ]

    print("🔍 Checking for already completed combinations...")
    completed_combinations = get_completed_combinations(db, all_rows)

    remaining_combinations = [
        params
        for params in valid_combinations
        if generate_run_name(*params) not in completed_combinations
    ]

    completed_count = len(valid_combinations) - len(remaining_combinations)
    print(
        f"Generated {len(valid_combinations)} valid combinations. "
        f"Found {completed_count} completed. Remaining: {len(remaining_combinations)}"
    )

    if not remaining_combinations:
        print("🎉 All combinations already processed!")
        return

    for params in tqdm(remaining_combinations, desc="Processing combinations"):
        (
            model_config,
            chunk_size,
            chunk_overlap,
            theme_name,
            chunking_strategy,
            similarity_metric,
        ) = params
        run_name = generate_run_name(*params)

        processed_files = db.get_processed_files(run_name)
        
        result = run_test(
            rows=all_rows,
            db=db,
            lance_db=lance_db,
            model_config=model_config,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            themes=GRID_SEARCH_PARAMS["themes"][theme_name],
            theme_name=theme_name,
            chunking_strategy=chunking_strategy,
            similarity_metric=similarity_metric,
            processed_files=processed_files,
            show_progress=False,
        )

        model_results = result.get("results", {})
        results_to_save = [
            (run_name, file_id, file_data)
            for file_id, file_data in model_results.get("files", {}).items()
            if file_data
        ]
        if results_to_save:
            db.add_model(
                run_name,
                model_config["name"],
                model_config["type"],
                chunk_size,
                chunk_overlap,
                theme_name,
                chunking_strategy,
                similarity_metric,
            )
            db.save_processing_results_batch(results_to_save)


def _aggregate_data(db: EmbeddingDatabase, lance_db: LanceDBClient, all_results: dict):
    """Aggregates data from results for reporting."""
    processed_data_for_interactive_page = {}
    all_models_metrics = {}
    model_embeddings_for_variance = {}

    for model_name, model_results in all_results.items():
        for file_id, file_data in model_results.get("files", {}).items():
            file_entry = processed_data_for_interactive_page.setdefault(
                file_id, {"embeddings": {}}
            )
            file_entry["embeddings"][model_name] = {
                "phrases": file_data.get("phrases", []),
                "similarities": file_data.get("similarities", []),
                "metrics": file_data.get("metrics", {}),
                "themes": file_data.get("themes", []),
            }

        model_info = db.get_model_info(model_name)
        if model_info:
            base_model_name = model_info["model_name"]
            table_name = f"embed_{base_model_name.replace('-', '_').replace('/', '_')}"
            
            try:
                table = lance_db.db.open_table(table_name)
                all_lance_records = table.to_pandas()
                current_model_embeddings = []
                for res in model_results.get("files", {}).values():
                    if res and "phrases" in res:
                        phrase_hashes = [get_text_hash(p) for p in res["phrases"]]
                        embeddings_df = all_lance_records[
                            all_lance_records["text_hash"].isin(phrase_hashes)
                        ]
                        if not embeddings_df.empty:
                            hash_to_vector = {
                                row["text_hash"]: row["vector"]
                                for _, row in embeddings_df.iterrows()
                            }
                            ordered_embeddings = [
                                hash_to_vector.get(h) for h in phrase_hashes
                            ]
                            ordered_embeddings = [
                                e for e in ordered_embeddings if e is not None
                            ]
                            if ordered_embeddings:
                                current_model_embeddings.append(
                                    np.array(ordered_embeddings)
                                )
                if current_model_embeddings:
                    model_embeddings_for_variance[
                        model_name
                    ] = current_model_embeddings
            except FileNotFoundError:
                pass  # Table might not exist for this model

        metrics_list = [
            res["metrics"] for res in model_results.get("files", {}).values() if res
        ]
        if metrics_list:
            avg_metrics = {
                key: float(np.mean([m[key] for m in metrics_list if key in m]))
                for key in metrics_list[0]
            }
            all_models_metrics[model_name] = avg_metrics
            db.add_evaluation_metrics(model_name, avg_metrics)

    return (
        processed_data_for_interactive_page,
        all_models_metrics,
        model_embeddings_for_variance,
    )


def generate_all_reports(db: EmbeddingDatabase, lance_db: LanceDBClient):
    """Generates all reports from the data in the database."""
    print("--- Generating All Reports ---")

    markdown_directory = os.path.join(os.path.dirname(__file__), "data", "markdown")
    all_rows = load_markdown_files(markdown_directory)
    file_metadata = {row[0]: {"nom": row[1], "type_lieu": row[2]} for row in all_rows}

    all_results = db.get_all_processing_results()
    if not all_results:
        print("No processing results found in the database. Run processing first.")
        return

    (
        processed_data_for_interactive_page,
        all_models_metrics,
        model_embeddings_for_variance,
    ) = _aggregate_data(db, lance_db, all_results)

    generate_main_page(
        {"files": processed_data_for_interactive_page},
        OUTPUT_DIR,
        len(all_results),
    )
    _generate_file_reports(db, all_results, file_metadata)
    _generate_global_reports(db, all_models_metrics, model_embeddings_for_variance)

    print(f"\n✅ All reports generated in '{OUTPUT_DIR}'.")


def _generate_file_reports(db, all_results, file_metadata):
    """Generates individual HTML and Markdown reports for each file."""
    for run_name, model_results in tqdm(
        all_results.items(), desc="Generating file reports"
    ):
        model_info = db.get_model_info(run_name)
        if not model_info:
            continue
        base_model_name = model_info["model_name"]

        for file_id, file_data in model_results.get("files", {}).items():
            metadata = file_metadata.get(
                file_id, {"nom": file_id, "type_lieu": "Unknown"}
            )
            if "phrases" in file_data and "similarities" in file_data:
                generate_heatmap_html(
                    identifiant=file_id,
                    nom=metadata["nom"],
                    type_lieu=metadata["type_lieu"],
                    themes=file_data.get("themes", []),
                    phrases=file_data["phrases"],
                    similarites_norm=np.array(file_data["similarities"]),
                    cmap=CMAP,
                    output_dir=OUTPUT_DIR,
                    model_name=base_model_name,
                    run_name=run_name,
                )


def _generate_global_reports(db, all_models_metrics, model_embeddings_for_variance):
    """Generates global comparison charts."""
    if all_models_metrics:
        plot_path = analyze_and_visualize_clustering_metrics(
            all_models_metrics, OUTPUT_DIR
        )
        if plot_path:
            db.add_global_chart("clustering_metrics", plot_path)

    if model_embeddings_for_variance:
        plot_path = analyze_and_visualize_variance(
            model_embeddings_for_variance, OUTPUT_DIR
        )
        if plot_path:
            db.add_global_chart("variance_analysis", plot_path)


def main():
    """Main function to manage the pipeline."""
    parser = argparse.ArgumentParser(description="Run embedding analysis and reporting.")
    parser.add_argument(
        "--generate-reports",
        action="store_true",
        help="Only generate reports from existing data.",
    )
    parser.add_argument(
        "--clear-db", action="store_true", help="Clear the database before running."
    )
    args = parser.parse_args()

    db = EmbeddingDatabase()
    lance_db = LanceDBClient()

    if args.clear_db:
        print("--- Clearing Databases ---")
        db.clear_database()
        # You might want to clear LanceDB as well, if applicable
        # For now, we assume it's managed separately or recreated as needed.

    if args.generate_reports:
        generate_all_reports(db, lance_db)
    else:
        run_processing(db, lance_db)
        generate_all_reports(db, lance_db)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
