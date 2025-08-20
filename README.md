# ForzaEmbed: Benchmarking Framework for Text Embeddings

ForzaEmbed is a Python framework for systematically benchmarking text embedding models and processing strategies. It performs an exhaustive grid search across a configurable parameter space to help you find the optimal configuration for your document corpus.

![Demo](docs/demo.gif)

## Table of Contents

- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [1. Installation](#1-installation)
  - [2. Place Your Documents](#2-place-your-documents)
  - [3. Configure the Analysis](#3-configure-the-analysis)
  - [4. Run the Pipeline](#4-run-the-pipeline)
- [Command-Line Usage](#command-line-usage)
  - [First Run](#first-run)
  - [Resuming a Run](#resuming-a-run)
  - [Generating Reports Only](#generating-reports-only)
- [Key Features](#key-features)
- [License](#license)

---

## How It Works

ForzaEmbed automates the process of evaluating text embedding configurations by following these steps:

1.  **Data Loading**: Ingests source documents from the `markdowns/` directory.
2.  **Grid Search**: Iterates through every combination of parameters defined in your configuration file (e.g., `configs/config.yml`).
3.  **Processing Pipeline**: For each combination, the framework:
    -   Chunks the text using a specified strategy.
    -   Generates embeddings using the selected model.
    -   Computes similarity scores based on defined themes.
4.  **Evaluation**: Calculates unsupervised metrics (like silhouette score and coherence) to assess the quality of the results.
5.  **Persistence & Caching**: Stores all results, metrics, and generated embeddings in a SQLite database. This caching accelerates subsequent runs by avoiding redundant computations.
6.  **Report Generation**: Produces detailed reports, including an interactive web interface, to visualize and analyze the findings.

---

## Project Structure

Understanding the directory layout is key to using ForzaEmbed effectively.

```
ForzaEmbed/
├── configs/
│   └── config.yml        # Your analysis configuration files go here.
├── markdowns/
│   └── document.md       # Your source text files (.md) go here.
├── reports/
│   └── ForzaEmbed_config.db # SQLite database for results.
├── src/
│   └── ...               # Source code of the application.
└── main.py               # The main script to run the tool.
```

-   **`configs/`**: This directory holds your YAML configuration files. You can create multiple configurations for different experiments.
-   **`markdowns/`**: Place the text documents you want to analyze here. The tool will process all `.md` files in this folder.
-   **`reports/`**: This is where all outputs are stored.
    -   **`ForzaEmbed_<config_name>.db`**: The central SQLite database. It stores all experiment results, metrics.

---

## Getting Started

### 1. Installation

This project uses `uv` for fast and efficient package management.

```bash
# 1. Clone the repository
git clone https://github.com/berangerthomas/ForzaEmbed.git
cd ForzaEmbed

# 2. Install dependencies
pip install uv
uv sync
```

### 2. Place Your Documents

Put your markdown (`.md`) files into the `markdowns/` directory.

### 3. Configure the Analysis

Open a configuration file (e.g., `configs/config.yml`) and define the parameters for your grid search. This includes:
-   Chunking strategies, sizes, and overlaps.
-   Embedding models to test (from Hugging Face, FastEmbed, etc.).
-   Similarity metrics.
-   Thematic keywords for the analysis.

Refer to the [Configuration Guide](#configuration-guide) below for detailed explanations.

### 4. Run the Pipeline

Execute the main script from your terminal to start the process. See the [Command-Line Usage](#command-line-usage) section below for detailed commands.

---

## Command-Line Usage

ForzaEmbed is controlled via a command-line interface.

### First Run

To start a new analysis from scratch, use the `--run` command and specify your configuration file.

```bash
python main.py --run --config-path configs/config.yml
```

This command will:
1.  Read the documents from the `markdowns/` directory (by default).
2.  Execute the grid search based on `configs/config.yml`.
3.  Save all results and embeddings to `reports/ForzaEmbed_config.db`.
4.  Generate an interactive report in the `reports/web/` directory.

### Resuming a Run

If a run is interrupted, simply execute the same command again. ForzaEmbed automatically detects completed work and resumes from where it left off.

```bash
python main.py --run --config-path configs/config.yml
```

### Generating Reports Only

If you want to regenerate the reports from existing data in the database without re-running the computations, use the `--generate-reports` command.

```bash
python main.py --generate-reports --config-path configs/config.yml
```

This is useful for changing the number of top results displayed (`--top-n`) or tweaking report settings.

---

## Configuration Guide

The `config.yml` file is the control center for your analysis. It's written in YAML and is divided into several sections. Here’s a breakdown of how to format it based on the standard configuration:

```yaml
# Parameters for the grid search
grid_search_params:
  chunk_size: [10, 20, 50, 100, 250, 500, 1000]
  chunk_overlap: [0, 5, 10, 25, 50, 100, 200]
  chunking_strategy: ["langchain", "raw", "semchunk", "nltk", "spacy"]
  similarity_metrics: ["cosine", "euclidean", "manhattan", "dot_product", "chebyshev"]
  themes:
    Theme_Name_1: ["keyword1", "keyword2", "related phrase 3"]
    Theme_Name_2: ["another keyword", "topic phrase 2"]

# Models to be tested in the grid search
models_to_test:
  - type: "fastembed"
    name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    dimensions: 384
  - type: "huggingface"
    name: "Qwen/Qwen3-Embedding-0.6B"
    dimensions: 1024
  - type: "api"
    name: "nomic-embed-text"
    base_url: "https://api.example.com/v1"
    dimensions: 768
    timeout: 240

# General settings
similarity_threshold: 0.6
output_dir: "reports"

# Database settings
database:
  intelligent_quantization: true

# Multiprocessing settings
multiprocessing:
  max_workers_api: 16
  max_workers_local: null # Set to a number to limit CPU cores for local models
  maxtasksperchild: 10
  embedding_batch_size_api: 100
  embedding_batch_size_local: 500
  file_batch_size: 50
  api_batch_sizes:
    mistral: 50
    default: 100
```

### `grid_search_params`

This section defines the parameter space for the grid search. The framework will test every possible combination of the values you provide.

-   `chunk_size`: A list of integers representing the different chunk sizes (in tokens or characters, depending on the strategy) to test.
-   `chunk_overlap`: A list of integers for the number of tokens/characters to overlap between chunks.
-   `chunking_strategy`: A list of chunking algorithms to evaluate.
-   `similarity_metrics`: A list of metrics for calculating similarity scores.
-   `themes`: A dictionary where each key is a theme name (e.g., `Economics_and_Finance`) and the value is a list of keywords and phrases related to that theme. The analysis will be based on these themes.

### `models_to_test`

A list of embedding models to evaluate. Each model is an object with the following properties:

-   `type`: The provider of the model. Can be `fastembed`, `huggingface`, `sentence_transformers`, or `api`.
-   `name`: The official model name (e.g., `"intfloat/multilingual-e5-large"`).
-   `dimensions`: The embedding dimension of the model.
-   `base_url` (for `api` type): The base URL of the embedding API endpoint.
-   `timeout` (for `api` type, optional): The request timeout in seconds.

### General & Database Settings

-   `similarity_threshold`: A float between 0.0 and 1.0. In the t-SNE visualization, points with a similarity score above this threshold will be highlighted.
-   `output_dir`: The directory where reports will be saved (default is `"reports"`).
-   `database.intelligent_quantization`: If `true`, enables optimizations to reduce the database size by storing numerical data in more efficient formats.

### `multiprocessing`

Configure settings for parallel processing to speed up computations.

-   `max_workers_api` / `max_workers_local`: The number of parallel workers for API-based and local models.
-   `embedding_batch_size_api` / `embedding_batch_size_local`: The number of texts to process in a single batch for embedding generation.

---

## Key Features

-   **Exhaustive Grid Search**: Systematically evaluates combinations of chunking strategies, chunk sizes, overlaps, embedding models, and similarity metrics.
-   **Broad Model Support**: Interfaces with multiple embedding providers, including local models (Hugging Face, FastEmbed, SentenceTransformers) and API-based services.
-   **Versatile Chunking**: Implements various chunking methods: `langchain`, `raw`, `semchunk`, `nltk`, and `spacy`.
-   **Multiple Similarity Metrics**: Supports `cosine`, `euclidean`, `manhattan`, `dot_product`, and `chebyshev`.
-   **Unsupervised Evaluation**: Integrates metrics like silhouette score, internal coherence, and local density to quantify embedding quality without needing labeled data.
-   **Resumable & Cached**: Caches embeddings in a SQLite database to accelerate subsequent runs and allows resuming interrupted workflows seamlessly.
-   **Rich Reporting**: Produces similarity heatmaps, CSV exports, and an interactive web interface for in-depth results analysis.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
