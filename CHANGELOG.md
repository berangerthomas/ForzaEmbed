# Changelog

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
