Installation
============

Requirements
------------

ForzaEmbed requires Python 3.13 or higher.

Installation
------------

This project uses **uv** for dependency management.

1. Clone the repository::

    git clone https://github.com/berangerthomas/ForzaEmbed.git
    cd ForzaEmbed

2. Install dependencies::

    uv sync

Verifying Installation
----------------------

To verify that ForzaEmbed is installed correctly, run::

    python -c "from src.core.core import ForzaEmbed; print('ForzaEmbed installed successfully!')"

Troubleshooting
---------------

spaCy Models
~~~~~~~~~~~~

If you plan to use spaCy chunking strategy, download the required language models::

    python -m spacy download en_core_web_sm  # English
    python -m spacy download fr_core_news_sm  # French

NLTK Data
~~~~~~~~~

NLTK data should be downloaded automatically. If not, run::

    python -m nltk.downloader punkt
