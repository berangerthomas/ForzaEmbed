import argparse
import itertools
import os

import numpy as np

from src.config import CMAP, GRID_SEARCH_PARAMS, MODELS_TO_TEST, OUTPUT_DIR
from src.data_loader import load_markdown_files
from src.database import EmbeddingDatabase
from src.processing import run_test
from src.config import SIMILARITY_THRESHOLD
from src.reporting import (
    analyze_and_visualize_clustering_metrics,
    analyze_and_visualize_variance,
    generate_explanatory_markdown,
    generate_filtered_markdown,
    generate_heatmap_html,
)
from src.web_generator import generate_main_page, generate_model_page


def run_processing(db: EmbeddingDatabase):
    """Exécute le traitement des données et sauvegarde les résultats dans la base de données."""
    print("--- Starting Data Processing ---")
    db.clear_database()

    markdown_directory = os.path.join(os.path.dirname(__file__), "data", "markdown")
    all_rows = load_markdown_files(markdown_directory)

    param_grid = {
        "model_config": MODELS_TO_TEST,
        "chunk_size": GRID_SEARCH_PARAMS["chunk_size"],
        "chunk_overlap": GRID_SEARCH_PARAMS["chunk_overlap"],
        "theme_name": list(GRID_SEARCH_PARAMS["themes"].keys()),
        "chunking_strategy": GRID_SEARCH_PARAMS["chunking_strategy"],
    }

    param_combinations = list(itertools.product(*param_grid.values()))

    for i, params in enumerate(param_combinations, 1):
        (
            model_config,
            chunk_size,
            chunk_overlap,
            theme_name,
            chunking_strategy,
        ) = params

        model_name = model_config["name"]
        theme_func = GRID_SEARCH_PARAMS["themes"][theme_name]
        themes = theme_func()

        run_name = f"{model_name}_cs{chunk_size}_co{chunk_overlap}_t{theme_name}_s{chunking_strategy}"
        print(f"\n--- Running Test {i}/{len(param_combinations)}: {run_name} ---")

        db.add_model(
            run_name,
            model_name,
            model_config["type"],
            chunk_size,
            chunk_overlap,
            theme_name,
            chunking_strategy,
        )

        model_results = run_test(
            all_rows,
            model_config,
            chunk_size,
            chunk_overlap,
            themes,
            OUTPUT_DIR,
            theme_name,
            chunking_strategy,
        )

        for file_id, file_data in model_results.get("files", {}).items():
            if file_data:
                db.save_processing_result(run_name, file_id, file_data)


def generate_all_reports(db: EmbeddingDatabase):
    """Génère tous les rapports à partir des données de la base de données."""
    print("--- Generating All Reports ---")

    # Load file metadata to get 'nom' and 'type_lieu'
    markdown_directory = os.path.join(os.path.dirname(__file__), "data", "markdown")
    all_rows = load_markdown_files(markdown_directory)
    file_metadata = {row[0]: {"nom": row[1], "type_lieu": row[2]} for row in all_rows}

    all_results = db.get_all_processing_results()
    if not all_results:
        print("No processing results found in the database. Run processing first.")
        return

    # --- Structures de données pour les rapports ---
    processed_data_for_interactive_page = {}
    all_models_metrics = {}
    model_embeddings_for_variance = {}
    all_labels_for_tsne = []
    first_run = True

    # --- Agrégation des données ---
    for model_name, model_results in all_results.items():
        # Pour la page interactive
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

        # Pour les rapports statiques
        current_model_embeddings = [
            res["embeddings_data"]["embeddings"]
            for res in model_results.get("files", {}).values()
            if res and "embeddings_data" in res
        ]
        current_model_labels = [
            res["embeddings_data"]["labels"]
            for res in model_results.get("files", {}).values()
            if res and "embeddings_data" in res
        ]

        if current_model_embeddings:
            model_embeddings_for_variance[model_name] = current_model_embeddings
        if first_run and current_model_labels:
            all_labels_for_tsne.extend(np.concatenate(current_model_labels))
            first_run = False

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

    # --- Génération des pages web ---
    print("\n--- Generating Web Pages ---")
    final_data_structure = {"files": processed_data_for_interactive_page}
    generate_main_page(final_data_structure, OUTPUT_DIR)

    for model_name, metrics in all_models_metrics.items():
        model_page_data = {"name": model_name, **metrics}
        generate_model_page(model_page_data, OUTPUT_DIR)

    # --- Génération des rapports individuels (HTML et Markdown) ---
    print("\n--- Generating Individual Reports (HTML & Markdown) ---")
    for run_name, model_results in all_results.items():
        model_info = db.get_model_info(run_name)
        if not model_info:
            continue
        base_model_name = model_info["model_name"]

        for file_id, file_data in model_results.get("files", {}).items():
            metadata = file_metadata.get(
                file_id, {"nom": file_id, "type_lieu": "Unknown"}
            )
            nom = metadata["nom"]
            type_lieu = metadata["type_lieu"]

            if "phrases" in file_data and "similarities" in file_data:
                similarities_np = np.array(file_data["similarities"])
                themes = file_data.get("themes", [])

                # Générer le heatmap HTML
                generate_heatmap_html(
                    identifiant=file_id,
                    nom=nom,
                    type_lieu=type_lieu,
                    themes=themes,
                    phrases=file_data["phrases"],
                    similarites_norm=similarities_np,
                    cmap=CMAP,
                    output_dir=OUTPUT_DIR,
                    model_name=base_model_name,
                    run_name=run_name,
                )

                # Générer le markdown filtré
                generate_filtered_markdown(
                    identifiant=file_id,
                    nom=nom,
                    type_lieu=type_lieu,
                    phrases=file_data["phrases"],
                    similarites_norm=similarities_np,
                    threshold=SIMILARITY_THRESHOLD,
                    output_dir=OUTPUT_DIR,
                    model_name=base_model_name,
                    run_name=run_name,
                )

                # Générer le markdown explicatif
                generate_explanatory_markdown(
                    identifiant=file_id,
                    nom=nom,
                    type_lieu=type_lieu,
                    phrases=file_data["phrases"],
                    similarites_norm=similarities_np,
                    themes=themes,
                    threshold=SIMILARITY_THRESHOLD,
                    output_dir=OUTPUT_DIR,
                    model_name=base_model_name,
                    run_name=run_name,
                )

    # --- Génération des rapports statiques ---
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

        # Since themes can change, we can't reliably generate a single t-SNE for all runs.
        # This part could be adapted to generate a t-SNE per theme set if needed.
        # For now, we disable it to avoid errors.
        # if all_labels_for_tsne:
        #     tsne_plots = generate_tsne_visualization(
        #         model_embeddings_for_variance, all_labels_for_tsne, list(GRID_SEARCH_PARAMS["themes"].keys())[0], OUTPUT_DIR
        #     )
        #     for model_name, path in tsne_plots.items():
        #         db.add_generated_file(model_name, "tsne_visualization", path)

    print(f"\n✅ All reports generated. Outputs are in the '{OUTPUT_DIR}' directory.")
    print(
        f"   Please open '{os.path.join(OUTPUT_DIR, 'index.html')}' to view the results."
    )


def main():
    """Fonction principale pour gérer le pipeline."""
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
