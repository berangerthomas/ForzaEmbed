import os
import pickle

from src.config import OUTPUT_DIR
from src.reporting import (
    analyze_and_visualize_clustering_metrics,
    analyze_and_visualize_variance,
)


def regenerate():
    """
    Régénère les graphiques de comparaison à partir des résultats sauvegardés.
    """
    print("--- Regenerating Comparison Plots ---")

    # --- Charger les résultats d'évaluation ---
    eval_results_path = os.path.join(OUTPUT_DIR, "evaluation_results.pkl")
    if os.path.exists(eval_results_path):
        with open(eval_results_path, "rb") as f:
            evaluation_results = pickle.load(f)
        print(f"Loaded evaluation results from {eval_results_path}")
        analyze_and_visualize_clustering_metrics(evaluation_results, OUTPUT_DIR)
    else:
        print(f"Error: Evaluation results file not found at {eval_results_path}")

    # --- Charger les embeddings pour l'analyse de variance ---
    variance_data_path = os.path.join(OUTPUT_DIR, "model_embeddings_for_variance.pkl")
    if os.path.exists(variance_data_path):
        with open(variance_data_path, "rb") as f:
            model_embeddings_for_variance = pickle.load(f)
        print(f"Loaded variance data from {variance_data_path}")
        analyze_and_visualize_variance(model_embeddings_for_variance, OUTPUT_DIR)
    else:
        print(f"Error: Variance data file not found at {variance_data_path}")

    print(f"\n✅ All plots regenerated in the '{OUTPUT_DIR}' directory.")


if __name__ == "__main__":
    regenerate()
