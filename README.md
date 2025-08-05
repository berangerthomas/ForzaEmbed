# ForzaEmbed

ForzaEmbed is a Python framework for benchmarking text embedding models and text processing strategies. It performs a grid search over a variety of parameters, including chunking strategies, embedding models, and similarity metrics, to find a suitable configuration for a given dataset.

![Demo](docs/demo.gif)

## Features

- **Grid Search:** Systematically tests combinations of chunking strategies, chunk sizes, overlaps, embedding models, and similarity metrics.
- **Model Support:** Supports various embedding models through a configuration file, including API-based services and local models from Hugging Face and FastEmbed.
- **Chunking Strategies:** Includes multiple chunking methods: `langchain`, `raw`, `semchunk`, `nltk`, and `spacy`.
- **Similarity Metrics:** Evaluates performance using `cosine`, `euclidean`, `manhattan`, `dot_product`, and `chebyshev` similarity metrics.
- **Caching:** Caches generated embeddings to accelerate subsequent runs.
- **Resumable Workflows:** Can resume interrupted grid searches.
- **Reporting:** Generates reports, including text heatmaps and CSV files, to visualize and compare the performance of different parameter combinations.
- **Command-Line Interface:** Provides a CLI to run the pipeline, manage the database, and generate reports.

## How It Works

ForzaEmbed follows a systematic process to evaluate embedding configurations:

1.  **Data Loading:** Loads text data from a directory of markdown files.
2.  **Grid Search:** Iterates through a predefined grid of parameters from the `config.yml` file.
3.  **Processing:** For each combination of parameters, the tool processes the text, generates embeddings, and calculates similarity scores.
4.  **Database Storage:** All results are stored in a SQLite database.
5.  **Report Generation:** After the grid search is complete, ForzaEmbed generates reports and visualizations.

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

ForzaEmbed is controlled via the command line.

### Running the Grid Search

To run the full grid search and reporting pipeline, use the `--run` flag:

```bash
python main.py --run
```

This command will resume the grid search if it was previously interrupted. To start from scratch, use the `--no-resume` flag.

### Generating Reports

To regenerate reports from a completed grid search, use the `--generate-reports` flag:

```bash
python main.py --generate-reports
```

You can limit the comparison charts to the top N models using the `--top-n` argument:

```bash
python main.py --generate-reports --top-n 10
```

### Command-Line Options

| Argument              | Description                                                              |
| --------------------- | ------------------------------------------------------------------------ |
| `--db-path`           | Path to the SQLite database file.                                        |
| `--config-path`       | Path to the YAML configuration file.                                     |
| `--data-source`       | Path to the directory containing markdown files.                         |
| `--run`               | Run the full grid search and reporting pipeline.                         |
| `--generate-reports`  | Only generate reports from existing data.                                |
| `--no-resume`         | Start the grid search from scratch.                                      |
| `--clear-db`          | Clear the main database before running.                                  |
| `--clear-cache`       | Clear the embedding cache before running.                                |
| `--top-n`             | Limit comparison charts to the top N models.                             |
| `--refresh-metrics`   | Refresh evaluation metrics for all existing runs.                        |

## Configuration

The behavior of ForzaEmbed is controlled by the `config.yml` file, which is divided into several sections:

-   **`grid_search_params`**: Defines the parameters for the grid search, such as `chunk_size`, `chunk_overlap`, `chunking_strategy`, and `similarity_metrics`.
-   **`models_to_test`**: A list of embedding models to be evaluated. You can specify the `type` (e.g., `api`, `fastembed`, `huggingface`), `name`, and other model-specific parameters.
-   **`general_settings`**: General configuration options, such as the `similarity_threshold` and `output_dir`.
-   **`multiprocessing`**: Settings to configure multiprocessing.

## Contributing

Contributions are welcome. If you have suggestions for improvements or find any issues, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
