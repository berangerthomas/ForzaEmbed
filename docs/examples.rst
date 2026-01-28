Examples
========

Common use cases and example configurations for ForzaEmbed.

Example 1: Finding Opening Hours
---------------------------------

This example shows how to configure ForzaEmbed to find opening hours in documents.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

    grid_search_params:
      chunk_size: [100, 250, 500]
      chunk_overlap: [10, 25]
      chunking_strategy: ["semchunk", "langchain"]
      similarity_metrics: ["cosine", "dot_product"]
      
      themes:
        opening_hours: [
          "opening hours",
          "schedule",
          "Monday to Friday",
          "open from",
          "closed on",
          "business hours"
        ]

    models_to_test:
      - type: "fastembed"
        name: "BAAI/bge-small-en-v1.5"
        dimensions: 384

    similarity_threshold: 0.6
    output_dir: "reports"

    database:
      intelligent_quantization: true

    multiprocessing:
      max_workers_local: 4
      embedding_batch_size_local: 500

Python Code
~~~~~~~~~~~

.. code-block:: python

    from src.core.core import ForzaEmbed

    # Initialize
    app = ForzaEmbed(
        db_path="reports/opening_hours.db",
        config_path="configs/opening_hours.yml"
    )

    # Run analysis on markdown files
    app.run_grid_search(data_source="markdowns/locations/")

    # Generate reports showing top 10 configurations
    app.generate_reports(top_n=10)

Example 2: Comparing Multiple Models
-------------------------------------

Systematically compare different embedding models.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

    grid_search_params:
      chunk_size: [250]
      chunk_overlap: [25]
      chunking_strategy: ["semchunk"]
      similarity_metrics: ["cosine", "dot_product"]
      
      themes:
        topic: [
          "artificial intelligence",
          "machine learning",
          "neural networks"
        ]

    models_to_test:
      - type: "fastembed"
        name: "BAAI/bge-small-en-v1.5"
        dimensions: 384
      
      - type: "fastembed"
        name: "BAAI/bge-base-en-v1.5"
        dimensions: 768
      
      - type: "sentence_transformers"
        name: "all-MiniLM-L6-v2"
        dimensions: 384
      
      - type: "sentence_transformers"
        name: "all-mpnet-base-v2"
        dimensions: 768

    similarity_threshold: 0.6
    output_dir: "reports"

Python Code
~~~~~~~~~~~

.. code-block:: python

    from src.core.core import ForzaEmbed

    app = ForzaEmbed(
        db_path="reports/model_comparison.db",
        config_path="configs/model_comparison.yml"
    )

    app.run_grid_search(data_source="markdowns/ai_papers/")
    
    # Show all combinations to compare models
    app.generate_reports(top_n=-1)

Example 3: Resume from Interrupted Run
---------------------------------------

ForzaEmbed automatically caches results, allowing you to resume interrupted runs.

.. code-block:: python

    from src.core.core import ForzaEmbed

    app = ForzaEmbed(
        db_path="reports/large_analysis.db",
        config_path="configs/config.yml"
    )

    # First run - processes all files
    app.run_grid_search(data_source="markdowns/", resume=True)

    # If interrupted, run again with same parameters
    # It will skip already processed combinations
    app.run_grid_search(data_source="markdowns/", resume=True)

Example 4: Generate Reports from Existing Database
---------------------------------------------------

Regenerate or modify reports without rerunning the analysis.

.. code-block:: python

    from src.core.core import ForzaEmbed

    app = ForzaEmbed(
        db_path="reports/existing_analysis.db",
        config_path="configs/config.yml"
    )

    # Don't run analysis, just regenerate reports
    # Show top 50 configurations
    app.generate_reports(top_n=50)

    # Generate single HTML file for all documents
    app.generate_reports(top_n=25, single_file=True)

Command Line Examples
---------------------

First run with full grid search::

    python main.py \\
        --config-path configs/config.yml \\
        --data-source markdowns/ \\
        --run

Generate reports only::

    python main.py \\
        --config-path configs/config.yml \\
        --generate-reports \\
        --top-n 25

Single HTML file for all documents::

    python main.py \\
        --config-path configs/config.yml \\
        --generate-reports \\
        --top-n 25 \\
        --single-file

Custom data source and output::

    python main.py \\
        --config-path configs/custom.yml \\
        --data-source data/documents/ \\
        --run \\
        --top-n 15

Performance Optimization Tips
------------------------------

Smart Grid Search Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ForzaEmbed automatically optimizes grid search by detecting chunking strategies that don't use chunk_size and chunk_overlap parameters.

**Parameter-Insensitive Strategies** (sentence-based):

* ``nltk``: Uses ``nltk.sent_tokenize()`` - ignores chunk parameters
* ``spacy``: Uses spaCy's sentence segmentation - ignores chunk parameters

**Parameter-Sensitive Strategies** (size-based):

* ``langchain``: Uses RecursiveCharacterTextSplitter with exact size control
* ``semchunk``: Semantic chunking with size limits
* ``raw``: Character-based chunking with overlap

**Impact Example:**

With a configuration having:

* 7 chunk_sizes × 7 chunk_overlaps = 35 valid pairs
* 5 strategies (3 sensitive + 2 insensitive)
* 6 models, 5 metrics, 3 themes

**Before optimization**: 15,750 combinations

**After optimization**: 9,630 combinations

**Result**: 38.9% reduction, 1.64x speedup

The system automatically uses only one chunk configuration for ``nltk`` and ``spacy`` since different sizes would produce identical results.

For Large Datasets
~~~~~~~~~~~~~~~~~~

1. **Start small**: Test with a subset first
2. **Use caching**: Enable ``intelligent_quantization``
3. **Parallel processing**: Increase ``max_workers_local``
4. **Batch processing**: Adjust ``embedding_batch_size_local``

.. code-block:: yaml

    multiprocessing:
      max_workers_local: 8
      embedding_batch_size_local: 1000
      file_batch_size: 100

For API-Based Models
~~~~~~~~~~~~~~~~~~~~

1. **Manage rate limits**: Adjust ``max_workers_api``
2. **Batch wisely**: Set appropriate ``api_batch_sizes``
3. **Handle retries**: Built-in retry logic handles temporary failures

.. code-block:: yaml

    multiprocessing:
      max_workers_api: 4
      api_batch_sizes:
        openai: 100
        mistral: 50

For Memory Constraints
~~~~~~~~~~~~~~~~~~~~~~

1. **Reduce batch sizes**
2. **Enable quantization**
3. **Process fewer files at once**

.. code-block:: yaml

    database:
      intelligent_quantization: true
    
    multiprocessing:
      file_batch_size: 10
      maxtasksperchild: 5
