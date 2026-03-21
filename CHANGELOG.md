# Changelog

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
