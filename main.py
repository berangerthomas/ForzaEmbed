import argparse
import itertools
import os

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
from src.processing import run_test
from src.reporting import (
    analyze_and_visualize_clustering_metrics,
    analyze_and_visualize_variance,
    generate_explanatory_markdown,
    generate_filtered_markdown,
    generate_heatmap_html,
)
from src.web_generator import generate_main_page


def run_processing(db: EmbeddingDatabase):
    """Runs data processing and saves results to the database."""
    print("--- Starting Data Processing ---")

    markdown_directory = os.path.join(os.path.dirname(__file__), "data", "markdown")
    all_rows = load_markdown_files(markdown_directory)

    param_grid = {
        "model_config": MODELS_TO_TEST,
        "chunk_size": GRID_SEARCH_PARAMS["chunk_size"],
        "chunk_overlap": GRID_SEARCH_PARAMS["chunk_overlap"],
        "theme_name": list(GRID_SEARCH_PARAMS["themes"].keys()),
        "chunking_strategy": GRID_SEARCH_PARAMS["chunking_strategy"],
        "similarity_metric": SIMILARITY_METRICS,
    }

    param_combinations = list(itertools.product(*param_grid.values()))

    # Global progress bar over all combinations
    for i, params in enumerate(
        tqdm(param_combinations, desc="Total runs", unit="run"), 1
    ):
        (
            model_config,
            chunk_size,
            chunk_overlap,
            theme_name,
            chunking_strategy,
            similarity_metric,
        ) = params

        model_name = model_config["name"]
        themes = GRID_SEARCH_PARAMS["themes"][theme_name]
        run_name = f"{model_name}_cs{chunk_size}_co{chunk_overlap}_t{theme_name}_s{chunking_strategy}_m{similarity_metric}"

        if db.model_exists(run_name):
            tqdm.write(
                f"--- Skipping Test {i}/{len(param_combinations)}: {run_name} (already exists) ---"
            )
            continue

        tqdm.write(f"--- Running Test {i}/{len(param_combinations)}: {run_name} ---")

        try:
            db.add_model(
                run_name,
                model_name,
                model_config["type"],
                chunk_size,
                chunk_overlap,
                theme_name,
                chunking_strategy,
                similarity_metric,
            )

            model_results = run_test(
                rows=all_rows,
                db=db,
                model_config=model_config,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                themes=themes,
                theme_name=theme_name,
                chunking_strategy=chunking_strategy,
                similarity_metric=similarity_metric,
                show_progress=False,
            )

            # Batch save results for the current run
            results_to_save = [
                (run_name, file_id, file_data)
                for file_id, file_data in model_results.get("files", {}).items()
                if file_data
            ]
            if results_to_save:
                db.save_processing_results_batch(results_to_save)

        except Exception as e:
            tqdm.write(
                f"❌ Error in Test {i}/{len(param_combinations)}: {run_name} ({e})"
            )
            continue


def _aggregate_data(db, all_results):
    """Aggregates data from results for reporting."""
    processed_data_for_interactive_page = {}
    all_models_metrics = {}
    model_embeddings_for_variance = {}

    for model_name, model_results in all_results.items():
        # For the interactive page
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

        # For static reports
        current_model_embeddings = [
            res["embeddings_data"]["embeddings"]
            for res in model_results.get("files", {}).values()
            if res and "embeddings_data" in res
        ]
        if current_model_embeddings:
            model_embeddings_for_variance[model_name] = current_model_embeddings

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


def _generate_file_reports(db, all_results, file_metadata):
    """Generates individual HTML and Markdown reports for each file."""
    print("\n--- Generating Individual Reports (HTML & Markdown) ---")
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
                generate_filtered_markdown(
                    identifiant=file_id,
                    nom=metadata["nom"],
                    type_lieu=metadata["type_lieu"],
                    phrases=file_data["phrases"],
                    similarites_norm=np.array(file_data["similarities"]),
                    threshold=SIMILARITY_THRESHOLD,
                    output_dir=OUTPUT_DIR,
                    model_name=base_model_name,
                    run_name=run_name,
                )
                generate_explanatory_markdown(
                    identifiant=file_id,
                    nom=metadata["nom"],
                    type_lieu=metadata["type_lieu"],
                    phrases=file_data["phrases"],
                    similarites_norm=np.array(file_data["similarities"]),
                    themes=file_data.get("themes", []),
                    threshold=SIMILARITY_THRESHOLD,
                    output_dir=OUTPUT_DIR,
                    model_name=base_model_name,
                    run_name=run_name,
                )


def _generate_global_reports(db, all_models_metrics, model_embeddings_for_variance):
    """Generates global comparison charts."""
    print("\n--- Generating Static Comparison Plots ---")
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


def generate_all_reports(db: EmbeddingDatabase):
    """Generates all reports from the data in the database."""
    print("--- Generating All Reports ---")

    # 1. Load metadata and results
    markdown_directory = os.path.join(os.path.dirname(__file__), "data", "markdown")
    all_rows = load_markdown_files(markdown_directory)
    file_metadata = {row[0]: {"nom": row[1], "type_lieu": row[2]} for row in all_rows}

    all_results = db.get_all_processing_results()
    if not all_results:
        print("No processing results found in the database. Run processing first.")
        return

    # 2. Aggregate data for reports
    (
        processed_data_for_interactive_page,
        all_models_metrics,
        model_embeddings_for_variance,
    ) = _aggregate_data(db, all_results)

    # 3. Generate web pages
    print("\n--- Generating Web Pages ---")
    final_data_structure = {"files": processed_data_for_interactive_page}
    generate_main_page(final_data_structure, OUTPUT_DIR)

    # 4. Generate individual file reports
    _generate_file_reports(db, all_results, file_metadata)

    # 5. Generate global comparison reports
    _generate_global_reports(db, all_models_metrics, model_embeddings_for_variance)

    print(f"\n✅ All reports generated. Outputs are in the '{OUTPUT_DIR}' directory.")
    print(
        f"   Please open '{os.path.join(OUTPUT_DIR, 'index.html')}' to view the results."
    )


def main():
    """Main function to manage the pipeline."""
    parser = argparse.ArgumentParser(
        description="Run embedding analysis and reporting."
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run the full pipeline: processing and reporting.",
    )
    parser.add_argument(
        "--generate-reports",
        action="store_true",
        help="Generate reports from existing data in the database.",
    )
    args = parser.parse_args()

    db = EmbeddingDatabase()

    if args.run_all:
        run_processing(db)
        generate_all_reports(db)
    elif args.generate_reports:
        generate_all_reports(db)
    else:
        print("No action specified. Use --run-all or --generate-reports.")
        print("Defaulting to --run-all.")
        run_processing(db)
        generate_all_reports(db)


if __name__ == "__main__":
    main()
