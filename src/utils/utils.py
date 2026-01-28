import re
import subprocess
import sys
from typing import Dict, List

import nltk
import semchunk
import spacy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from spacy.language import Language

# --- spaCy model loading ---
SPACY_MODELS: Dict[str, Language] = {}


def get_spacy_model(language: str) -> Language:
    """
    Loads and caches a spaCy model for a given language.
    Downloads the model if it's not available.
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
    text: str, chunk_size: int, chunk_overlap: int, **kwargs
) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(text)


def _chunk_semchunk(text: str, chunk_size: int, **kwargs) -> List[str]:
    return list(
        semchunk.chunk(  # type: ignore
            text,
            chunk_size=chunk_size,
            token_counter=lambda text: len(text.split()),
            offsets=False,
        )
    )


def _chunk_nltk(text: str, language: str = "fr", **kwargs) -> List[str]:
    lang_map = {"fr": "french", "en": "english"}
    nltk_lang = lang_map.get(language, "french")
    return nltk.sent_tokenize(text, language=nltk_lang)


def _chunk_spacy(text: str, language: str = "fr", **kwargs) -> List[str]:
    nlp = get_spacy_model(language)
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


def _chunk_raw(text: str, chunk_size: int, chunk_overlap: int, **kwargs) -> List[str]:
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

CHUNKING_STRATEGIES = {
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
    """
    Splits the text into segments using a specified strategy.

    Args:
        text (str): Text to split.
        chunk_size (int): Size of the chunks.
        chunk_overlap (int): Overlap between chunks.
        strategy (str): The chunking strategy to use.
        language (str): The language of the text ('fr' or 'en').

    Returns:
        List[str]: List of extracted segments.
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


# Checks if a text contains a pattern related to opening hours.
def contains_horaire_pattern(text: str, keywords: dict) -> bool:
    """
    Checks if the text contains opening hours patterns.

    Args:
        text (str): Text to analyze.
        keywords (dict): Dictionary of keywords for the regex.

    Returns:
        bool: True if a pattern is found, otherwise False.
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


# Extracts the context around a target sentence in a list of sentences.
def extract_context_around_phrase(phrases: list[str], phrase_index: int) -> str:
    """
    Extracts and highlights the context around a target sentence.

    Args:
        phrases (list[str]): List of sentences.
        phrase_index (int): Index of the target sentence.

    Returns:
        str: Context with the target sentence highlighted.
    """
    if 0 <= phrase_index < len(phrases):
        return f"**{phrases[phrase_index].strip()}**"
    return ""
