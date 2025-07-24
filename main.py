import os

import numpy as np

from src.config import MODELS_TO_TEST, OUTPUT_DIR
from src.data_loader import load_markdown_files
from src.evaluation_metrics import (
    calculate_clustering_metrics,
    calculate_cohesion_separation,
)
from src.processing import run_test
from src.reporting import (
    analyze_and_visualize_clustering_metrics,
    analyze_and_visualize_variance,
)


def main():
    """
    Fonction principale pour exécuter l'analyse des embeddings.
    """
    # --- Data Loading ---
    markdown_directory = os.path.join(os.path.dirname(__file__), "data", "markdown")
    all_rows = load_markdown_files(markdown_directory)
    test_rows = all_rows

    # --- Run Processing ---
    model_embeddings_for_variance = {}
    evaluation_results = {}

    for config in MODELS_TO_TEST:
        model_name = config["name"]
        embeddings_list, labels_list = run_test(test_rows, config, OUTPUT_DIR)

        if embeddings_list and labels_list:
            model_embeddings_for_variance[model_name] = embeddings_list

            all_embeddings = np.vstack(
                [e for e in embeddings_list if e is not None and e.size > 0]
            )
            all_labels = np.concatenate(
                [l for l in labels_list if l is not None and l.size > 0]
            )

            if all_embeddings.size > 0 and all_labels.size > 0:
                print(f"\n--- Calculating Clustering Metrics for {model_name} ---")
                cohesion_sep_metrics = calculate_cohesion_separation(
                    all_embeddings, all_labels
                )
                clustering_metrics = calculate_clustering_metrics(
                    all_embeddings, all_labels
                )
                evaluation_results[model_name] = {
                    **cohesion_sep_metrics,
                    **clustering_metrics,
                }
                print(f"  - Results: {evaluation_results[model_name]}")
            else:
                print(
                    f"No valid embeddings or labels for {model_name}, skipping metrics."
                )

    # --- Generate Comparison Plots ---
    print("\n--- Generating Final Comparison Plots ---")
    if evaluation_results:
        analyze_and_visualize_clustering_metrics(evaluation_results, OUTPUT_DIR)
    else:
        print("No evaluation results to plot.")

    if model_embeddings_for_variance:
        analyze_and_visualize_variance(model_embeddings_for_variance, OUTPUT_DIR)
    else:
        print("\nNo embeddings were generated. Skipping variance analysis.")

    print(f"\n✅ All processing complete. Outputs are in the '{OUTPUT_DIR}' directory.")


if __name__ == "__main__":
    main()
