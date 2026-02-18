"""Text processing utilities for ForzaEmbed.

This module provides utility functions for text chunking, pattern matching,
and context extraction. It supports multiple chunking strategies including
langchain, semchunk, nltk, spacy, and raw character-based chunking.

Example:
    Chunk text using different strategies::

        from src.utils.utils import chunk_text

        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50, strategy="langchain")
"""

import re
import subprocess
import sys
from typing import Callable, Dict, List

import nltk
import semchunk
import spacy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from spacy.language import Language

# --- spaCy model loading ---
SPACY_MODELS: Dict[str, Language] = {}


def get_spacy_model(language: str) -> Language:
    """Load and cache a spaCy model for a given language.

    Downloads the model if it's not available locally.

    Args:
        language: Language code ('fr' for French, 'en' for English).

    Returns:
        Loaded spaCy Language model.

    Raises:
        ValueError: If the language is not supported.
    """
    model_map = {
        "fr": "fr_core_news_sm",
        "en": "en_core_web_sm",
    }
    model_name = model_map.get(language)
    if not model_name:
        raise ValueError(f"Unsupported language for spaCy: {language}")

    if language not in SPACY_MODELS:
        try:
            SPACY_MODELS[language] = spacy.load(model_name)
        except OSError:
            print(f"Downloading spaCy model for '{language}' ({model_name})...")
            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", model_name]
            )
            SPACY_MODELS[language] = spacy.load(model_name)
    return SPACY_MODELS[language]

# Download nltk data if not already present
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# --- Helper functions for each chunking strategy ---


def _chunk_langchain(
    text: str, chunk_size: int, chunk_overlap: int, **kwargs: str
) -> List[str]:
    """Chunk text using LangChain's RecursiveCharacterTextSplitter.

    Args:
        text: Text to chunk.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Overlap between consecutive chunks.
        **kwargs: Additional arguments (unused).

    Returns:
        List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(text)


def _chunk_semchunk(text: str, chunk_size: int, **kwargs: str) -> List[str]:
    """Chunk text using semantic chunking.

    Args:
        text: Text to chunk.
        chunk_size: Target chunk size in tokens.
        **kwargs: Additional arguments (unused).

    Returns:
        List of text chunks.
    """
    return list(
        semchunk.chunk(  # type: ignore
            text,
            chunk_size=chunk_size,
            token_counter=lambda text: len(text.split()),
            offsets=False,
        )
    )


def _chunk_nltk(text: str, language: str = "fr", **kwargs: str) -> List[str]:
    """Chunk text using NLTK sentence tokenization.

    Note:
        This strategy ignores chunk_size and chunk_overlap parameters.

    Args:
        text: Text to chunk.
        language: Language code for tokenization.
        **kwargs: Additional arguments (unused).

    Returns:
        List of sentences as chunks.
    """
    lang_map = {"fr": "french", "en": "english"}
    nltk_lang = lang_map.get(language, "french")
    return nltk.sent_tokenize(text, language=nltk_lang)


def _chunk_spacy(text: str, language: str = "fr", **kwargs: str) -> List[str]:
    """Chunk text using spaCy sentence segmentation.

    Note:
        This strategy ignores chunk_size and chunk_overlap parameters.

    Args:
        text: Text to chunk.
        language: Language code for the spaCy model.
        **kwargs: Additional arguments (unused).

    Returns:
        List of sentences as chunks.
    """
    nlp = get_spacy_model(language)
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


def _chunk_raw(text: str, chunk_size: int, chunk_overlap: int, **kwargs: str) -> List[str]:
    """Chunk text using raw character-based splitting.

    Args:
        text: Text to chunk.
        chunk_size: Size of each chunk in characters.
        chunk_overlap: Overlap between consecutive chunks.
        **kwargs: Additional arguments (unused).

    Returns:
        List of text chunks.

    Raises:
        ValueError: If chunk_size <= 0, chunk_overlap < 0, or chunk_size <= chunk_overlap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_size <= chunk_overlap:
        raise ValueError(
            "chunk_size must be greater than chunk_overlap for raw chunking."
        )

    step = chunk_size - chunk_overlap
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]


# --- Main chunking function using a dictionary-based approach ---

ChunkingFunc = Callable[..., List[str]]

CHUNKING_STRATEGIES: Dict[str, ChunkingFunc] = {
    "langchain": _chunk_langchain,
    "semchunk": _chunk_semchunk,
    "nltk": _chunk_nltk,
    "spacy": _chunk_spacy,
    "raw": _chunk_raw,
}


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    strategy: str = "langchain",
    language: str = "fr",
) -> List[str]:
    """Split text into segments using a specified strategy.

    Supports multiple chunking strategies with different characteristics.
    Some strategies (nltk, spacy) ignore chunk_size and chunk_overlap.

    Args:
        text: Text to split.
        chunk_size: Size of chunks in characters (ignored by nltk, spacy).
        chunk_overlap: Overlap between chunks (ignored by nltk, spacy).
        strategy: Chunking strategy to use. One of: 'langchain', 'semchunk',
            'nltk', 'spacy', 'raw'.
        language: Language of the text ('fr' or 'en').

    Returns:
        List of extracted text segments.

    Raises:
        ValueError: If an unknown chunking strategy is specified.
    """
    if strategy not in CHUNKING_STRATEGIES:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    # Call the appropriate chunking function
    chunking_func = CHUNKING_STRATEGIES[strategy]
    chunks = chunking_func(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        language=language,
    )

    # Post-process the chunks
    if strategy == "raw":
        # For raw strategy, keep chunks as they are to preserve exact size, but filter out empty strings
        return [str(chunk) for chunk in chunks if isinstance(chunk, str) and chunk]
    else:
        # For other strategies, strip whitespace and filter out empty chunks
        return [
            str(chunk).strip()
            for chunk in chunks
            if isinstance(chunk, str) and chunk.strip()
        ]


def contains_horaire_pattern(text: str, keywords: dict[str, list[str]]) -> bool:
    """Check if text contains patterns related to opening hours.

    Uses regex patterns to detect time-related content including days,
    times, and action keywords.

    Args:
        text: Text to analyze.
        keywords: Dictionary with 'jours' (days) and 'actions' keyword lists.

    Returns:
        True if an opening hours pattern is found, False otherwise.
    """
    # Build regex patterns from the keywords dictionary
    time_pattern = r"\d{1,2}h(\d{2})?"
    days_pattern = r"\b(" + "|".join(keywords["jours"]) + r")\b"
    keyword_pattern = r"\b(" + "|".join(keywords["actions"]) + r")\b"
    range_pattern = r"\d{1,2}h(\d{2})?\s*[-\/]\s*\d{1,2}h(\d{2})?"

    # Combine all patterns into a single regex for efficiency
    combined_pattern = "|".join(
        [time_pattern, days_pattern, keyword_pattern, range_pattern]
    )

    # Check if the combined pattern is found
    if re.search(combined_pattern, text, re.IGNORECASE):
        return True

    return False


def extract_context_around_phrase(phrases: list[str], phrase_index: int) -> str:
    """Extract and highlight context around a target sentence.

    Args:
        phrases: List of sentences.
        phrase_index: Index of the target sentence.

    Returns:
        The target sentence wrapped in markdown bold formatting,
        or empty string if index is out of bounds.
    """
    if 0 <= phrase_index < len(phrases):
        return f"**{phrases[phrase_index].strip()}**"
    return ""
