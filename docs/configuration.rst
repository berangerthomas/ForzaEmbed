Configuration Guide
===================

ForzaEmbed uses YAML configuration files to define grid search parameters and settings.

Configuration File Structure
-----------------------------

A typical configuration file has four main sections:

.. code-block:: yaml

    grid_search_params:
      # Parameters to test in grid search
      
    models_to_test:
      # Embedding models to evaluate
      
    # General settings
    output_dir: "reports"
    
    database:
      # Database optimization settings
      
    multiprocessing:
      # Performance tuning

Grid Search Parameters
----------------------

chunk_size
~~~~~~~~~~

List of chunk sizes (in characters) to test.

.. code-block:: yaml

    chunk_size: [100, 250, 500, 1000]

* **Smaller values** (50-200): Better for fine-grained analysis, more chunks
* **Medium values** (200-500): Balanced approach
* **Larger values** (500-2000): Capture more context, fewer chunks

chunk_overlap
~~~~~~~~~~~~~

Overlap between consecutive chunks (prevents splitting related content).

.. code-block:: yaml

    chunk_overlap: [0, 10, 25, 50]

* **0**: No overlap
* **10-25**: Recommended for most cases
* **50+**: High overlap, useful for ensuring context continuity

chunking_strategy
~~~~~~~~~~~~~~~~~

Method used to split text into chunks.

.. code-block:: yaml

    chunking_strategy: ["langchain", "semchunk", "nltk", "spacy", "raw"]

Available strategies:

* **langchain**: Recursive character-based splitting
* **semchunk**: Semantic chunking that respects sentence boundaries
* **nltk**: Sentence tokenization using NLTK
* **spacy**: Advanced NLP-based segmentation
* **raw**: Simple character-based splitting

similarity_metrics
~~~~~~~~~~~~~~~~~~

Distance/similarity metrics for comparing embeddings.

.. code-block:: yaml

    similarity_metrics: ["cosine", "dot_product", "euclidean", "manhattan", "chebyshev"]

* **cosine**: Measures angle between vectors (default, normalized)
* **dot_product**: Combines angle and magnitude
* **euclidean**: Straight-line distance
* **manhattan**: Sum of absolute differences
* **chebyshev**: Maximum difference along any dimension

themes
~~~~~~

Define semantic themes for filtering relevant chunks.

.. code-block:: yaml

    themes:
      schedule:
        - "opening hours"
        - "schedule"
        - "Monday to Friday"
        - "closed on weekends"
      
      location:
        - "address"
        - "located at"
        - "you can find us"

Each theme is a list of keywords/phrases that define what to look for.

Models Configuration
--------------------

FastEmbed Models
~~~~~~~~~~~~~~~~

.. code-block:: yaml

    - type: "fastembed"
      name: "BAAI/bge-small-en-v1.5"
      dimensions: 384

Popular FastEmbed models:

* ``BAAI/bge-small-en-v1.5`` (384D): Fast, good quality
* ``BAAI/bge-base-en-v1.5`` (768D): Better quality, slower
* ``nomic-ai/nomic-embed-text-v1.5`` (768D): Strong performance

Sentence Transformers
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    - type: "sentence_transformers"
      name: "all-MiniLM-L6-v2"
      dimensions: 384

Popular models:

* ``all-MiniLM-L6-v2`` (384D): Fast and efficient
* ``all-mpnet-base-v2`` (768D): High quality
* ``paraphrase-multilingual-mpnet-base-v2`` (768D): Multilingual

Hugging Face Transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    - type: "transformers"
      name: "jinaai/jina-embeddings-v3"
      dimensions: 1024

API-based Models
~~~~~~~~~~~~~~~~

.. code-block:: yaml

    - type: "api"
      name: "text-embedding-3-small"
      dimensions: 1536
      base_url: "https://api.openai.com/v1"
      timeout: 30

Requires setting environment variables::

    export OPENAI_API_KEY="your-api-key"


General Settings
----------------

output_dir
~~~~~~~~~~

Directory for saving reports and databases.

.. code-block:: yaml

  output_dir: "reports"

Note
~~~~

The server-side `similarity_threshold` configuration has been removed. The
interactive HTML report includes a client-side similarity slider that
reclassifies and highlights points in the visualization; use that slider to
adjust thresholding at view time.

generate_filtered_markdowns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate filtered markdown files containing only relevant chunks.

.. code-block:: yaml

    generate_filtered_markdowns: false

Database Settings
-----------------

intelligent_quantization
~~~~~~~~~~~~~~~~~~~~~~~~

Compress data to reduce database size.

.. code-block:: yaml

    database:
      intelligent_quantization: true

* **true**: Compress floats to 16-bit, reduce storage by ~50%
* **false**: Full 64-bit precision

Performance Settings
--------------------

.. code-block:: yaml

    multiprocessing:
      max_workers_api: 16
      max_workers_local: null  # Auto-detect CPU cores
      maxtasksperchild: 10
      embedding_batch_size_api: 100
      embedding_batch_size_local: 500
      file_batch_size: 50
      
      api_batch_sizes:
        mistral: 50
        openai: 100
        voyage: 100
        default: 100

* **max_workers_api**: Parallel API calls
* **max_workers_local**: Parallel local computations
* **maxtasksperchild**: Restart workers after N tasks (memory management)
* **embedding_batch_size_***: Batch size for embedding computation
* **file_batch_size**: Number of files processed in parallel

Complete Example
----------------

.. code-block:: yaml

    grid_search_params:
      chunk_size: [100, 250, 500]
      chunk_overlap: [0, 10, 25]
      chunking_strategy: ["langchain", "semchunk"]
      similarity_metrics: ["cosine", "dot_product"]
      
      themes:
        hours:
          - "opening hours"
          - "schedule"
          - "Monday"
          - "closed"

    models_to_test:
      - type: "fastembed"
        name: "BAAI/bge-small-en-v1.5"
        dimensions: 384

    output_dir: "reports"
    generate_filtered_markdowns: false

    database:
      intelligent_quantization: true

    multiprocessing:
      max_workers_api: 8
      max_workers_local: null
      maxtasksperchild: 10
      embedding_batch_size_api: 100
      embedding_batch_size_local: 500
      file_batch_size: 50
      api_batch_sizes:
        default: 100
