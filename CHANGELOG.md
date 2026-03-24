# Changelog

## [1.4.0] - 2026-03-24

### Added
- **Multi-projection support (UMAP / t-SNE / PCA)**: `src/services/visualization_service.py` now exposes `get_or_create_projections`, which computes and caches multiple projection methods (per-method cache keys) and returns a dictionary of projection results with method-specific metadata (e.g. KL divergence, perplexity, iterations, explained variance).
 - **Heatmap thresholding slider (textual)**: Add a global similarity `threshold` slider to the interactive HTML report that now controls both the scatter plot and the textual similarity heatmap. Chunks with similarity below the threshold are visually dimmed in the heatmap (`.chunk--dimmed`). The slider value defaults to `0.00` and is preserved when switching files/models.
 - **Floating threshold control**: The similarity threshold control is now available as a draggable vertical floating slider positioned on the left side of the report. The floating control syncs with the scatterplot threshold slider, persists its position in `localStorage`, and dimms low-similarity chunks in the textual heatmap.

### Changed
- **Frontend reporting**: `src/reporting/templates/main.js` and `src/reporting/templates/template.html` updated to allow selecting between t-SNE, UMAP and PCA from the UI; show projection metadata (KL, perplexity, iterations, explained variance); display chunk counts; handle active projection switching and adapt scatter plotting to different projection payload shapes. Adds `activeProjection` state and helper functions (e.g. `getTSNEColor`) to support multi-projection rendering.
- **Processing API call**: `src/core/processing.py` updated to call `get_or_create_projections` (generalized multi-method API) instead of the previous t-SNE-only helper.

### Removed
- **Server-side similarity threshold**: `similarity_threshold` has been removed from the application configuration and server-side processing. Server-side generation of similarity-filtered markdowns has been disabled; visualization thresholding is now handled client-side via the interactive slider in the HTML report.
- **Environment variable**: `SIMILARITY_THRESHOLD` removed from environment configuration.

### Fixed / Improved
- **Resilient SentenceTransformer loading**: `src/clients/sentencetransformers_client.py` now retries model instantiation with `trust_remote_code=True` on failure to improve robustness when loading remote models.
- **Safer projection handling**: visualization service includes fallbacks and stricter type handling to avoid runtime errors for missing metadata or unexpected payload shapes.

## [1.3.0] - 2026-03-21

### Fixed
- **GPU caching**: Removed permanent GPU/CPU fallback caching in FastEmbed client. GPU detection now re-attempts on each call for dynamic GPU availability. Added `reset_instance()` method for manual model reload.

### Added
- **Visualization**: Added a continuous color gradient legend for textual similarity in the HTML report.
- **Token length protection**: New `max_tokens` parameter in `ModelConfig` for FastEmbed and SentenceTransformers. Texts exceeding this limit are now split into smaller chunks and recombined using pooling.
- **Dynamic embedding chunking**: When `max_tokens` is set, long texts are automatically split into chunks that fit within the token limit, and their embeddings are combined using a pooling strategy. Four strategies available:
  - `max` (default): Max pooling - captures most salient features
  - `average`: Mean pooling - preserves overall semantic content
  - `weighted`: Weighted pooling - gives more importance to first chunks
  - `last`: Uses only the last chunk - useful for summaries/conclusions
- **SentenceTransformers batch size**: Added `batch_size` parameter to control memory usage.
- **Quantization toggle**: New `quantize_metrics` option in database settings. Set to `false` to store metrics in full float32 precision without quantization loss.
- **New utility module**: `src/utils/embedding_pooling.py` provides `split_text_into_chunks()` and `pool_embeddings()` functions.

### Changed
- **Visualization**: Unified the t-SNE scatter plot point colors with the textual similarity heatmap palette.
- **Visualization**: The threshold slider now greys out points below the threshold (instead of assigning them a specific color) to better highlight relevant chunks.
- **Visualization**: The threshold slider's value is now preserved across parameter switches in the UI, and defaults to `0.00`.

## [1.2.0] - 2026-02-18

### Added
- **Threshold slider**: interactive similarity threshold slider on the t-SNE scatter plot - reclassifies points instantly (above/below threshold).
- **Externalized templates**: report templates (`template.html`, `style.css`, `main.js`, `worker.js`) moved to `src/reporting/templates/`; `web_generator.py` reads them dynamically at build time.

### Changed
- **Report aesthetics**: redesign of the HTML report stylesheet.
- **Type annotations**: type hints added across all modules.
- **Docstrings**: Google/Sphinx-style docstrings.


## [1.1.0] - 2026-01-27

### Added
- **Interactive Tooltips**: Help tooltips on report page sliders and individual slider values (chunking strategies, similarity metrics, theme keywords) to guide users.

### Changed
- **Fast CLI Help**: Lazy loading of heavy dependencies (torch, transformers, etc.) for instant `--help` response.

### Documentation
- **Sphinx Documentation**: API documentation covering all modules (core, clients, services, metrics, reporting, utils).
- **CI/CD**: GitHub Actions workflow for automated documentation deployment to GitHub Pages.


## [1.0.1] - 2026-01-27

### Added
- **Automated releases**: GitHub Actions workflow for automatic versioning and releases based on CHANGELOG.md entries.


## [1.0.0] - 2026-01-17

### Added
- **Core Framework**: Complete pipeline for text embedding model evaluation and comparison.
- **Grid Search Engine**: Systematic testing of parameter combinations (chunk sizes, overlaps, strategies, models) with resumption capabilities.
- **Multi-Provider Support**: 
    - Local: FastEmbed (CPU/GPU), SentenceTransformers, Hugging Face Transformers.
    - API: Generic API client structure for remote services.
- **Chunking Strategies**: Integration with LangChain, SemChunk, NLTK, spaCy, and raw text processing.
- **Similarity Metrics**: Support for Cosine, Euclidean, Manhattan, Dot Product, and Chebyshev distances.
- **Persistent Caching**: SQLite storage for embeddings and results to avoid redundant computations.
- **Quantization**: Database optimization to reduce storage footprint while maintaining accuracy.
- **Reporting System**: Generation of standalone, interactive HTML reports with embedded visualizations.
- **Cluster Analysis**: Silhouette score calculation with detailed intra/inter-cluster distance decomposition.
- **Theme Analysis**: Configuration-based thematic evaluation of embeddings.
- **CLI Interface**: Command-line tool main.py for running pipelines and generating reports.
