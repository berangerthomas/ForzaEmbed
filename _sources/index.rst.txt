.. ForzaEmbed documentation master file

Welcome to ForzaEmbed's documentation!
=======================================

**ForzaEmbed** is a Python framework for systematically benchmarking text embedding models and processing strategies. 
It performs an exhaustive grid search across a configurable parameter space to help you find the optimal configuration 
for your document corpus.

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/Python-3.13+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

Key Features
------------

* **Automated Grid Search**: Test all combinations of chunk sizes, overlap, chunking strategies, similarity metrics, and embedding models
* **Standalone Interactive Visualization**: Single-file HTML report with embedded data for visualizing embedding similarities directly on text (no server required)
* **Multiple Embedding Models**: Support for FastEmbed, Sentence Transformers, Hugging Face, and API-based models
* **Flexible Chunking**: Compare different text segmentation strategies (LangChain, SemChunk, NLTK, spaCy, raw)
* **Comprehensive Metrics**: Silhouette analysis with intra/inter-cluster distance decomposition
* **Caching**: SQLite-based caching to avoid redundant computations
* **Performance Tracking**: Measure and compare embedding computation time across configurations

Quick Start
-----------

Installation::

    git clone https://github.com/berangerthomas/ForzaEmbed.git
    cd ForzaEmbed
    uv sync

Basic Usage::

    from src.core.core import ForzaEmbed
    
    # Initialize ForzaEmbed
    app = ForzaEmbed(
        db_path="reports/my_analysis.db",
        config_path="configs/config.yml"
    )
    
    # Run grid search
    app.run_grid_search(data_source="markdowns/")
    
    # Generate reports
    app.generate_reports(top_n=25)

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   configuration
   examples
   optimization

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/clients
   api/services
   api/metrics
   api/reporting
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
