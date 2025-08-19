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
│   ├── ForzaEmbed_config.db # SQLite database for results and cache.
│   └── web/                # Output directory for generated reports.
├── src/
│   └── ...               # Source code of the application.
└── main.py               # The main script to run the tool.
```

-   **`configs/`**: This directory holds your YAML configuration files. You can create multiple configurations for different experiments.
-   **`markdowns/`**: Place the text documents you want to analyze here. The tool will process all `.md` files in this folder.
-   **`reports/`**: This is where all outputs are stored.
    -   **`ForzaEmbed_<config_name>.db`**: The central SQLite database. It stores all experiment results, metrics, and serves as the **cache for embeddings**. Deleting this file will erase your results and cache.
    -   **`web/`**: The interactive HTML reports are generated in this sub-directory.

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
