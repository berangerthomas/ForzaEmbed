# ForzaEmbed: Benchmarking Framework for Text Embeddings

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://berangerthomas.github.io/ForzaEmbed/)
[![Hugging Face Demo](https://img.shields.io/badge/🤗-Demo-yellow.svg)](https://huggingface.co/spaces/berangerthomas/forzaembeddemo)
[![GitHub release](https://img.shields.io/github/v/release/berangerthomas/ForzaEmbed)](https://github.com/berangerthomas/ForzaEmbed/releases)

ForzaEmbed is a Python framework for **benchmarking text embedding models** and processing strategies.

It runs a grid search over configurable hyperparameters (embedding model, chunking strategy, chunk size, similarity metric, etc.) and produces a **textual heatmap** highlighting theme-relevant text regions, alongside **t-SNE, UMAP, and PCA visualizations** to analyze embedding structure. The generated standalone HTML report is interactive: you can switch between projection methods and use a draggable floating vertical similarity-threshold slider; chunks and scatter points below the threshold are dimmed.

📖 **[Documentation](https://berangerthomas.github.io/ForzaEmbed/)** · 🚀 **[Live Demo](https://huggingface.co/spaces/berangerthomas/forzaembeddemo)** · 📦 **[Releases](https://github.com/berangerthomas/ForzaEmbed/releases)**

<!-- demo video -->
https://github.com/user-attachments/assets/74e2b6a6-db18-4a25-ba2a-8c6047552942

---

## How It Works

You drop your `.md` documents into `markdowns/`, define the parameter space in a YAML config file, and run `main.py`. ForzaEmbed then:

1. reads all documents from `markdowns/`;
2. expands the config into every combination of chunk size, overlap, chunking strategy, embedding model, and similarity metric;
3. for each combination: chunks the text, generates embeddings, and scores chunks against your defined themes;
4. evaluates each configuration using silhouette score (with intra/inter-cluster decomposition) and embedding computation time;
5. caches all results and embeddings in a SQLite database — completed combinations are skipped on subsequent runs;
6. generates a standalone interactive HTML report (heatmaps, t-SNE/UMAP/PCA visualizations, CSV exports) in `reports/`. The report includes UI controls for selecting projection method and a draggable floating similarity-threshold slider; chunks and scatter points below the threshold are dimmed.

> **Note on chunking strategies**: `langchain`, `raw`, and `semchunk` are parameter-sensitive (they use `chunk_size` and `chunk_overlap`). `nltk` and `spacy` are sentence-based and ignore those parameters — ForzaEmbed avoids generating redundant combinations for them, which can reduce the total number of runs by up to 40%.

---

## Project Structure

```
ForzaEmbed/
├── configs/          # YAML configuration files
├── docs/             # Documentation source (GitHub Pages)
├── markdowns/        # Source .md documents to analyse
├── reports/          # Generated reports and SQLite databases
├── src/              # Application source code
├── main.py           # Entry point
└── pyproject.toml    # Project metadata and dependencies
```

Each config run produces a dedicated database file: `reports/ForzaEmbed_<config_name>.db`.

---

## Getting Started

### 1. Installation

```bash
# Install uv (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows: winget install --id=astral-sh.uv -e

# Clone and install
git clone https://github.com/berangerthomas/ForzaEmbed.git
cd ForzaEmbed
uv sync
```

### 2. Add your documents

Put your `.md` files into `markdowns/`.

### 3. Configure and run

Edit `configs/config.yml` (see [Configuration Guide](#configuration-guide) below), then:

```bash
python main.py --run --config-path configs/config.yml
```

To reproduce the Hugging Face demo page locally, run:

```bash
uv run .\main.py --run --config-path configs/chicago.yml
```

Use the supplied `configs/chicago.yml` and place the provided `chicago.md` file into the `markdowns/` directory before running.

---

## Command-Line Usage

### First run

```bash
python main.py --run --config-path configs/config.yml
```

Reads documents from `markdowns/`, runs the grid search, saves results to `reports/ForzaEmbed_config.db`, and generates `reports/config_index.html`.

### Resuming an interrupted run

Re-run the same command. Completed combinations are detected and skipped automatically.

### Regenerating reports only

To rebuild reports from existing database data (e.g. to change `--top-n`) without rerunning computations:

```bash
python main.py --generate-reports --config-path configs/config.yml
```

---

## Configuration Guide

Below is an annotated example. For the full parameter reference, see the [documentation](https://berangerthomas.github.io/ForzaEmbed/).

```yaml
grid_search_params:
  chunk_size: [50, 100, 250, 500]
  chunk_overlap: [10, 25, 50]
  chunking_strategy: ["langchain", "raw", "semchunk", "nltk", "spacy"]
  similarity_metrics: ["cosine", "euclidean", "dot_product"]
  themes:
    opening_hours: ["opening hours", "public reception hours"]
    closing_days: ["closing day", "exceptional closure", "public holidays"]
    weekdays: ["monday", "tuesday", "wednesday", "thursday", "friday"]

models_to_test:
  - type: "fastembed"
    name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    dimensions: 384
  - type: "huggingface"
    name: "Qwen/Qwen3-Embedding-0.6B"
    dimensions: 1024
  - type: "api"
    name: "nomic-embed-text"
    base_url: "https://api.nomic.ai/v1"  # replace with your provider
    dimensions: 768
    timeout: 240

output_dir: "reports"

database:
  intelligent_quantization: true  # stores embeddings as float16 to reduce DB size

multiprocessing:
  max_workers_api: 16
  max_workers_local: null  # null = all available cores
  embedding_batch_size_api: 100
  embedding_batch_size_local: 500
```

Supported model types: `fastembed`, `huggingface`, `sentence_transformers`, `api`.  
Supported similarity metrics: `cosine`, `euclidean`, `manhattan`, `dot_product`, `chebyshev`.

---

## License

MIT — see [LICENSE](LICENSE).
