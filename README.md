# ForzaEmbed

ForzaEmbed is a Python framework designed for the systematic benchmarking of text embedding models and processing strategies. It performs an exhaustive grid search across a configurable parameter space to identify optimal configurations for a given document corpus.

![Demo](docs/demo.gif)

## Features

- **Grid Search:** Systematically evaluates combinations of chunking strategies, chunk sizes, overlaps, embedding models, and similarity metrics.
- **Model Support:** Interfaces with multiple embedding model providers via a configuration file, including API-based services and local models (Hugging Face, FastEmbed, SentenceTransformers).
- **Chunking Strategies:** Implements various chunking methods: `langchain`, `raw`, `semchunk`, `nltk`, and `spacy`.
- **Similarity Metrics:** Utilizes `cosine`, `euclidean`, `manhattan`, `dot_product`, and `chebyshev` for similarity calculations.
- **Evaluation Metrics:** The framework currently integrates a suite of unsupervised evaluation metrics to quantify the quality of embeddings and semantic clusters (e.g., silhouette score, internal coherence, local density index). The implementation of supervised metrics, which require ground-truth data, is planned for future development.
- **Caching:** Caches generated embeddings to accelerate subsequent runs.
- **Resumable Workflows:** Can resume an interrupted grid search from its last completed state.
- **Reporting:** Produces a range of outputs, including similarity heatmaps, CSV exports, and an interactive web interface for detailed results analysis.

## Workflow

ForzaEmbed follows a structured process to evaluate embedding configurations:

1.  **Data Loading:** Ingests source documents from a specified directory.
2.  **Grid Search Execution:** Iterates through the parameter grid defined in `config.yml`.
3.  **Processing Pipeline:** For each parameter combination, the framework chunks the text, generates embeddings, and computes similarity scores.
4.  **Evaluation:** Calculates performance metrics for the current configuration.
5.  **Results Persistence:** Stores all generated metrics and results in a SQLite database.
6.  **Report Generation:** Creates visualizations and analytical reports from the stored data.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/berangerthomas/ForzaEmbed.git
    cd ForzaEmbed
    ```

2.  **Install dependencies:**
    This project uses `uv` for package management.
    ```bash
    pip install uv
    uv sync
    ```

## Usage

ForzaEmbed is operated through its command-line interface (CLI).

### Running the Full Pipeline

To execute the entire grid search and report generation process, use the `--run` argument:

```bash
python main.py --run
```

By default, this command resumes an interrupted search. To start a new search from the beginning, add the `--no-resume` flag.

### Generating Reports Only

To regenerate reports from existing data in the database without re-running computations:

```bash
python main.py --generate-reports
```

### Command-Line Options

| Argument | Description |
| :--- | :--- |
| `--db-path` | Path to the SQLite database file. |
| `--config-path` | Path to the YAML configuration file. |
| `--data-source` | Path to the directory containing source documents. |
| `--run` | Executes the full grid search and reporting pipeline. |
| `--generate-reports` | Generates reports from existing data only. |
| `--no-resume` | Starts the grid search from scratch, ignoring any previous state. |
| `--clear-db` | Clears the main database before running. |
| `--clear-cache` | Clears the embedding cache before running. |
| `--all-combinations` | Generate charts with all combinations instead of the default top 25. |
| `--single-file` | Generate a single HTML report file for all documents. |

## Configuration

The framework's behavior is controlled by `config.yml`.

-   **`grid_search_params`**: Defines the hyperparameter space for the search, including `chunk_size`, `chunk_overlap`, `chunking_strategy`, and `similarity_metrics`.
-   **`models_to_test`**: A list of embedding models to evaluate. Specify the `type` (`api`, `fastembed`, `huggingface`), `name`, and other model-specific parameters.
-   **`general_settings`**: General configuration options, such as `similarity_threshold` and the `output_dir`.
-   **`multiprocessing`**: Settings to configure parallel processing.

## Contributing

Contributions are welcome. For suggestions or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
