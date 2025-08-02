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
from src.processing import run_test
from src.reporting import (
    analyze_and_visualize_clustering_metrics,
    analyze_and_visualize_variance,
    generate_explanatory_markdown,
    generate_filtered_markdown,
    generate_heatmap_html,
)
from src.web_generator import generate_main_page


def initialize_worker():
    """Initialize worker process with its own database connection."""
    global worker_db
    worker_db = EmbeddingDatabase()


def get_completed_combinations(db: EmbeddingDatabase, all_rows: list) -> set[str]:
    """
    Récupère la liste des combinaisons déjà complètement traitées.
    Une combinaison est considérée comme complète si tous les fichiers ont été traités
    avec des similarités calculées (pas seulement les embeddings).
    """
    completed_combinations = set()

    # Récupérer tous les run_names existants
    all_run_names = db.get_all_run_names()

    total_files = len(all_rows)

    for run_name in all_run_names:
        # Vérifier si ce run a traité tous les fichiers avec des similarités
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
    """Génère le nom de run standardisé pour une combinaison de paramètres."""
    model_name = model_config["name"]
    dimensions = model_config.get("dimensions", "auto")
    return f"{model_name}_d{dimensions}_cs{chunk_size}_co{chunk_overlap}_t{theme_name}_s{chunking_strategy}_m{similarity_metric}"


def process_combination_optimized(params_with_info):
    """
    Optimized version of process_combination with better error handling and resource management.
    """
    i, total, all_rows, params = params_with_info
    (
        model_config,
        chunk_size,
        chunk_overlap,
        theme_name,
        chunking_strategy,
        similarity_metric,
    ) = params

    # Use the worker's database connection
    global worker_db
    if "worker_db" not in globals():
        worker_db = EmbeddingDatabase()

    run_name = generate_run_name(
        model_config,
        chunk_size,
        chunk_overlap,
        theme_name,
        chunking_strategy,
        similarity_metric,
    )
    themes = GRID_SEARCH_PARAMS["themes"][theme_name]

    start_time = time.time()

    try:
        # Double-check: cette combinaison pourrait avoir été complétée par un autre processus
        processed_files_with_similarities = (
            worker_db.get_processed_files_with_similarities(run_name)
        )
        if len(processed_files_with_similarities) == len(all_rows):
            return {
                "status": "skipped",
                "run_name": run_name,
                "message": f"Skipped (completed by another process): {run_name}",
                "processing_time": 0,
            }

        # Vérifier les fichiers avec embeddings mais sans similarités
        processed_files = worker_db.get_processed_files(run_name)

        # Main processing
        worker_db.add_model(
            run_name,
            model_config["name"],
            model_config["type"],
            chunk_size,
            chunk_overlap,
            theme_name,
            chunking_strategy,
            similarity_metric,
        )

        model_results = run_test(
            rows=all_rows,
            db=worker_db,
            model_config=model_config,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            themes=themes,
            theme_name=theme_name,
            chunking_strategy=chunking_strategy,
            similarity_metric=similarity_metric,
            processed_files=processed_files,
            show_progress=False,
        )

        results_to_save = [
            (run_name, file_id, file_data)
            for file_id, file_data in model_results.get("files", {}).items()
            if file_data
        ]

        if results_to_save:
            worker_db.save_processing_results_batch(results_to_save)

        processing_time = time.time() - start_time
        return {
            "status": "success",
            "run_name": run_name,
            "message": f"Successfully processed: {run_name}",
            "processing_time": processing_time,
            "files_processed": len(results_to_save),
        }

    except Exception as e:
        processing_time = time.time() - start_time
        return {
            "status": "error",
            "run_name": run_name,
            "message": f"Error processing {run_name}: {e}",
            "processing_time": processing_time,
            "error": str(e),
        }


def calculate_optimal_processes():
    """Calculate optimal number of processes based on system resources."""
    cpu_count = multiprocessing.cpu_count()
    # Garder un coeur de libre pour que le PC reste utilisable
    return max(1, cpu_count - 1)


def run_processing(db: EmbeddingDatabase):
    """Enhanced multiprocessing data processing with better resource management."""
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
        params
        for params in param_combinations
        if params[1] > params[2]  # chunk_size > chunk_overlap
    ]

    print("🔍 Checking for already completed combinations...")
    completed_combinations = get_completed_combinations(db, all_rows)

    # Filter out already completed combinations
    remaining_combinations = []
    for params in valid_combinations:
        (
            model_config,
            chunk_size,
            chunk_overlap,
            theme_name,
            chunking_strategy,
            similarity_metric,
        ) = params
        run_name = generate_run_name(
            model_config,
            chunk_size,
            chunk_overlap,
            theme_name,
            chunking_strategy,
            similarity_metric,
        )
        if run_name not in completed_combinations:
            remaining_combinations.append(params)

    completed_count = len(valid_combinations) - len(remaining_combinations)

    print(
        f"Generated {len(param_combinations)} parameter combinations. "
        f"Filtered down to {len(valid_combinations)} valid combinations."
    )
    print(
        f"Found {completed_count} already completed combinations. "
        f"Remaining to process: {len(remaining_combinations)}"
    )

    if not remaining_combinations:
        print("🎉 All combinations already processed! Nothing to do.")
        return

    # Separate combinations: API models run sequentially, local models run in parallel
    api_combinations = [
        p for p in remaining_combinations if p[0].get("type") == "api"
    ]
    local_combinations = [
        p for p in remaining_combinations if p[0].get("type") != "api"
    ]

    print(
        f"Separated into {len(api_combinations)} API combinations (sequential) "
        f"and {len(local_combinations)} local combinations (parallel)."
    )

    # Initialize statistics
    successful_runs = 0
    skipped_runs = 0
    failed_runs = 0
    total_processing_time = 0

    # --- 1. Process API combinations sequentially ---
    if api_combinations:
        print("\n--- Running API tests sequentially to avoid rate limiting ---")
        global worker_db
        worker_db = db

        total_api_runs = len(api_combinations)
        api_tasks = [
            (i, total_api_runs, all_rows, params)
            for i, params in enumerate(api_combinations, 1)
        ]

        with tqdm(
            total=total_api_runs, desc="Processing API combinations", unit="combo"
        ) as pbar:
            for task in api_tasks:
                result = process_combination_optimized(task)
                status = result["status"]
                processing_time = result["processing_time"]
                run_name = result["run_name"]
                total_processing_time += processing_time

                if status == "success":
                    successful_runs += 1
                    files_processed = result.get("files_processed", 0)
                    pbar.set_description(f"✅ {run_name}")
                    pbar.set_postfix(
                        {
                            "Success": successful_runs,
                            "Skipped": skipped_runs,
                            "Failed": failed_runs,
                            "Files": files_processed,
                            "Time": f"{processing_time:.1f}s",
                        }
                    )
                elif status == "skipped":
                    skipped_runs += 1
                    pbar.set_description(f"⏭️ {run_name}")
                    pbar.set_postfix(
                        {
                            "Success": successful_runs,
                            "Skipped": skipped_runs,
                            "Failed": failed_runs,
                        }
                    )
                else:  # error
                    failed_runs += 1
                    pbar.set_description(f"❌ {run_name}")
                    pbar.set_postfix(
                        {
                            "Success": successful_runs,
                            "Skipped": skipped_runs,
                            "Failed": failed_runs,
                            "Error": result.get("error", "Unknown")[:30] + "...",
                        }
                    )
                    pbar.write(f"❌ ERROR: {result['message']}")
                pbar.update(1)

    # --- 2. Process local combinations in parallel ---
    if local_combinations:
        num_processes = calculate_optimal_processes()
        print(
            f"\n--- Running {len(local_combinations)} local tests using {num_processes} parallel processes ---"
        )
        total_local_runs = len(local_combinations)
        local_tasks = [
            (i, total_local_runs, all_rows, params)
            for i, params in enumerate(local_combinations, 1)
        ]

        try:
            with multiprocessing.Pool(
                processes=num_processes,
                initializer=initialize_worker,
                maxtasksperchild=10,
            ) as pool:
                results_iterator = pool.imap_unordered(
                    process_combination_optimized, local_tasks, chunksize=1
                )
                with tqdm(
                    total=total_local_runs,
                    desc="Processing local combinations",
                    unit="combo",
                ) as pbar:
                    for result in results_iterator:
                        status = result["status"]
                        processing_time = result["processing_time"]
                        run_name = result["run_name"]
                        total_processing_time += processing_time

                        if status == "success":
                            successful_runs += 1
                            files_processed = result.get("files_processed", 0)
                            pbar.set_description(f"✅ {run_name}")
                            pbar.set_postfix(
                                {
                                    "Success": successful_runs,
                                    "Skipped": skipped_runs,
                                    "Failed": failed_runs,
                                    "Files": files_processed,
                                    "Time": f"{processing_time:.1f}s",
                                }
                            )
                        elif status == "skipped":
                            skipped_runs += 1
                            pbar.set_description(f"⏭️ {run_name}")
                            pbar.set_postfix(
                                {
                                    "Success": successful_runs,
                                    "Skipped": skipped_runs,
                                    "Failed": failed_runs,
                                }
                            )
                        else:  # error
                            failed_runs += 1
                            pbar.set_description(f"❌ {run_name}")
                            pbar.set_postfix(
                                {
                                    "Success": successful_runs,
                                    "Skipped": skipped_runs,
                                    "Failed": failed_runs,
                                    "Error": result.get("error", "Unknown")[:30]
                                    + "...",
                                }
                            )
                            pbar.write(f"❌ ERROR: {result['message']}")
                        pbar.update(1)
        except KeyboardInterrupt:
            tqdm.write("\n🛑 Processing interrupted by user")
            return
        except Exception as e:
            tqdm.write(f"\n❌ Fatal error in multiprocessing: {e}")
            return

    # Final statistics
    avg_time = total_processing_time / max(1, successful_runs + failed_runs)
    tqdm.write("\n📊 Processing Summary:")
    tqdm.write(f"   ✅ Successful: {successful_runs}")
    tqdm.write(f"   ⏭️  Skipped: {skipped_runs}")
    tqdm.write(f"   ❌ Failed: {failed_runs}")
    tqdm.write(f"   🔄 Previously completed: {completed_count}")
    tqdm.write(f"   ⏱️  Average time per combination: {avg_time:.1f}s")
    tqdm.write(f"   🕒 Total processing time: {total_processing_time:.1f}s")


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

    # Count the number of combinations that have actually been processed
    processed_combinations_count = len(all_results)

    # 2. Aggregate data for reports
    (
        processed_data_for_interactive_page,
        all_models_metrics,
        model_embeddings_for_variance,
    ) = _aggregate_data(db, all_results)

    # 3. Generate web pages
    print("\n--- Generating Web Pages ---")
    final_data_structure = {"files": processed_data_for_interactive_page}
    generate_main_page(
        final_data_structure, OUTPUT_DIR, processed_combinations_count
    )

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
        description="Run embedding analysis and reporting. By default, it runs the full pipeline."
    )
    parser.add_argument(
        "--generate-reports",
        action="store_true",
        help="Only generate reports from existing data in the database, without processing.",
    )
    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Clear the database before running the full pipeline.",
    )
    args = parser.parse_args()

    db = EmbeddingDatabase()

    if args.clear_db:
        print("--- Clearing Database ---")
        db.clear_database()
        db.init_database()
        # Proceed to run the full pipeline after clearing
        run_processing(db)
        generate_all_reports(db)

    elif args.generate_reports:
        # Only generate reports
        generate_all_reports(db)

    else:
        # Default behavior: run the full pipeline without clearing
        run_processing(db)
        generate_all_reports(db)


if __name__ == "__main__":
    # This is required for multiprocessing on some platforms (like Windows)
    multiprocessing.freeze_support()
    main()
