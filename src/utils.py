import re
from typing import List

import nltk
import numpy as np
import semchunk
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter

# téléchargement du modèle de langue français
# python -m spacy download fr_core_news_sm

# Load spacy model
nlp = spacy.load("fr_core_news_sm")

# Download nltk data if not already present
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:  # type: ignore
    nltk.download("punkt")


# Splits a text into sentences or short segments.
def chunk_text(
    text: str, chunk_size: int, chunk_overlap: int, strategy: str = "langchain"
) -> List[str]:
    """
    Splits the text into segments of defined size and overlap.

    Args:
        text (str): Text to split.
        chunk_size (int): Size of the chunks. For nltk and spacy, this is ignored.
        chunk_overlap (int): Overlap between chunks. For nltk and spacy, this is ignored.
        strategy (str): 'langchain' for smart splitting, 'raw' for basic splitting,
                        'semchunk', 'nltk', or 'spacy' for other methods.

    Returns:
        List[str]: List of extracted segments.
    """
    if strategy == "langchain":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
    elif strategy == "semchunk":
        # semchunk uses chunk_size, but not chunk_overlap in the same way as 'raw'
        chunks = list(semchunk.chunk(  # type: ignore
            text,
            chunk_size=chunk_size,
            token_counter=lambda text: len(text.split()),
            offsets=False,
        ))
    elif strategy == "nltk":
        # nltk.sent_tokenize does not use chunk_size or chunk_overlap
        chunks = nltk.sent_tokenize(text, language="french")
    elif strategy == "spacy":
        # Spacy's sentence splitter does not use chunk_size or chunk_overlap
        doc = nlp(text)
        chunks = [sent.text for sent in doc.sents]
    elif strategy == "raw":
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_size == chunk_overlap:
            raise ValueError(
                "chunk_size and chunk_overlap must not be equal for raw chunking (step would be zero)."
            )
        step = chunk_size - chunk_overlap
        chunks = []
        for i in range(0, len(text), step):
            chunks.append(text[i : i + chunk_size])
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    return [str(chunk).strip() for chunk in chunks if str(chunk).strip()]


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
    context_window = 0
    start_idx = max(0, phrase_index - context_window)
    end_idx = min(len(phrases), phrase_index + context_window + 1)
    context_phrases = phrases[start_idx:end_idx]

    context_with_highlight = []
    for i, phrase in enumerate(context_phrases):
        actual_idx = start_idx + i
        if actual_idx == phrase_index:
            context_with_highlight.append(f"**{phrase.strip()}**")
        else:
            context_with_highlight.append(phrase.strip())
    return " ".join(context_with_highlight)
