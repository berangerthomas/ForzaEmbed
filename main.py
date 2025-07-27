import argparse
import os
import numpy as np

from src.config import BASE_THEMES, MODELS_TO_TEST, OUTPUT_DIR
from src.data_loader import load_markdown_files
from src.database import EmbeddingDatabase
from src.processing import run_test
from src.reporting import (
    analyze_and_visualize_clustering_metrics,
    analyze_and_visualize_variance,
    generate_tsne_visualization,
)
from src.web_generator import generate_main_page, generate_model_page


def run_processing(db: EmbeddingDatabase):
    """Exécute le traitement des données et sauvegarde les résultats dans la base de données."""
    print("--- Starting Data Processing ---")
    db.clear_database()
    
    markdown_directory = os.path.join(os.path.dirname(__file__), "data", "markdown")
    all_rows = load_markdown_files(markdown_directory)

    for config in MODELS_TO_TEST:
        model_name = config["name"]
        db.add_model(model_name, config["type"])
        
        model_results = run_test(all_rows, config, OUTPUT_DIR)

        for file_id, file_data in model_results.get("files", {}).items():
            if file_data:
                db.save_processing_result(model_name, file_id, file_data)


def generate_all_reports(db: EmbeddingDatabase):
    """Génère tous les rapports à partir des données de la base de données."""
    print("--- Generating All Reports ---")
    
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
            file_entry = processed_data_for_interactive_page.setdefault(file_id, {"phrases": None, "embeddings": {}})
            if file_entry.get("phrases") is None:
                file_entry["phrases"] = file_data.get("phrases", [])
            file_entry["embeddings"][model_name] = {
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
        
        metrics_list = [res["metrics"] for res in model_results.get("files", {}).values() if res]
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

    # --- Génération des rapports statiques ---
    print("\n--- Generating Static Comparison Plots ---")
    if all_models_metrics:
        plot_path = analyze_and_visualize_clustering_metrics(all_models_metrics, OUTPUT_DIR)
        if plot_path: db.add_global_chart("clustering_metrics", plot_path)
    
    if model_embeddings_for_variance:
        plot_path = analyze_and_visualize_variance(model_embeddings_for_variance, OUTPUT_DIR)
        if plot_path: db.add_global_chart("variance_analysis", plot_path)
        
        if all_labels_for_tsne:
            tsne_plots = generate_tsne_visualization(
                model_embeddings_for_variance, all_labels_for_tsne, BASE_THEMES, OUTPUT_DIR
            )
            for model_name, path in tsne_plots.items():
                db.add_generated_file(model_name, "tsne_visualization", path)

    print(f"\n✅ All reports generated. Outputs are in the '{OUTPUT_DIR}' directory.")
    print(f"   Please open '{os.path.join(OUTPUT_DIR, 'index.html')}' to view the results.")


def main():
    """Fonction principale pour gérer le pipeline."""
    parser = argparse.ArgumentParser(description="Run embedding analysis and reporting.")
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
