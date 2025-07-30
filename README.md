# ForzaEmbed

**Empower Your AI with the Strongest Embeds.**

---

## Overview

**ForzaEmbed** is a modular and high-performance Python framework for evaluating, comparing, and visualizing text embedding models. It is designed for real-world document analysis, providing a full, persistent pipeline from data ingestion to interactive web-based exploration and static reporting.

The entire process is optimized for speed and cost-efficiency, using an intelligent caching system and batch processing to avoid redundant computations and API calls.

---

## Key Features

- **High-Performance Caching:**  
  An integrated SQLite database caches all phrase embeddings. Once an embedding is computed, it's stored forever, drastically reducing processing time and API costs on subsequent runs.
- **Efficient Batch Processing:**  
  API calls for new embeddings and database writes are batched, minimizing network latency and I/O overhead for maximum throughput.
- **Persistent & Incremental Runs:**  
  All test configurations and results are stored in the database. The pipeline automatically skips previously completed runs, allowing you to add new models or parameters and only process what's new.
- **Flexible Embedding Model Support:**  
  Plug-and-play with local models (e.g., SentenceTransformers), FastEmbed, or remote API-based models (OpenAI, Mistral, VoyageAI).
- **Configurable Analysis:**  
  Tune chunking strategies, chunk size/overlap, and similarity metrics (cosine, euclidean, manhattan) through a central configuration.
- **Theme-based Semantic Analysis:**  
  Define custom "themes" (semantic queries) to automatically categorize and extract relevant information from documents.
- **Rich, Interactive Dashboard:**  
  Generates a comprehensive HTML dashboard to explore results, compare model metrics, and visualize similarity heatmaps.
- **Automated Reporting:**  
  Produces detailed Markdown reports and static comparison plots (radar, bar, variance analysis).

---

## Architecture & Workflow

The project is built around a simple, powerful workflow orchestrated by a central SQLite database.

```
1. Input
   - data/markdown/*.md

2. Processing (`python main.py --run-all`)
   - Chunks text from Markdown files.
   - Looks up chunk embeddings in the cache (ForzaEmbed.db).
   - Batch-embeds any new, uncached chunks via an embedding client.
   - Caches the new embeddings in the database.
   - Calculates similarity, runs analysis, and batch-saves all results to the database.

3. Reporting (`python main.py --generate-reports`)
   - Loads all processed results from the database.
   - Generates the interactive dashboard, static plots, and Markdown reports.

4. Outputs
   - data/heatmaps/ForzaEmbed.db  (The central database with all results)
   - data/heatmaps/index.html     (The main interactive dashboard)
   - data/heatmaps/...            (All other generated reports and plots)
```

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/berangerthomas/ForzaEmbed.git
    cd ForzaEmbed
    ```

2.  **Install dependencies:**  
    This project uses `uv` for fast dependency management.
    ```bash
    pip install uv
    uv sync
    ```

3.  **Configure API Keys:**  
    If you plan to use API-based models, create a `.env` file in the root directory by copying the example:
    ```bash
    cp .env.example .env
    ```
    Then, add your API keys to the `.env` file.

---

## Usage

### 1. Add Your Data
Place your Markdown files (`.md`) in the `data/markdown/` directory.

### 2. Run the Full Pipeline
Execute the main script. On the first run, this will process all documents with all configured models.

```bash
python main.py --run-all
```

On subsequent executions, the script will automatically skip any configurations that have already been run and are present in the database, only processing new ones.

### 3. Regenerate Reports
If you want to regenerate all reports from the existing data in the database without re-processing, use:
```bash
python main.py --generate-reports
```

### 4. Explore the Results
Open the generated dashboard in your browser:
```
data/heatmaps/index.html
```

### Resetting the Project
To start from scratch, simply delete the database file and run the pipeline again:
```bash
rm data/heatmaps/ForzaEmbed.db
```

---

## Configuration

Almost all configuration is centralized in `src/config.py`:

-   **Models:** Add or remove models to test in `MODELS_TO_TEST`.
-   **Themes:** Define semantic search themes in `GRID_SEARCH_PARAMS["themes"]`.
-   **Grid Search:** Tune chunk size, overlap, and similarity metrics in `GRID_SEARCH_PARAMS`.
-   **API Keys:** Manage API keys in the `.env` file.

---

## Contributing

Contributions are welcome! Please follow the standard fork-and-pull-request workflow. Before submitting, ensure your code is formatted with Ruff:

```bash
ruff format .
```

---

## License

This project is licensed under the GNU GENERAL PUBLIC License. See the [LICENSE](LICENSE) file for details.
