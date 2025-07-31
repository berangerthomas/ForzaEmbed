import re
from typing import List

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Splits a text into sentences or short segments.
def chunk_text(
    text: str, chunk_size: int, chunk_overlap: int, strategy: str = "langchain"
) -> List[str]:
    """
    Splits the text into segments of defined size and overlap.

    Args:
        text (str): Text to split.
        chunk_size (int): Size of the chunks.
        chunk_overlap (int): Overlap between chunks.
        strategy (str): 'langchain' for smart splitting, 'raw' for basic splitting.

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
    else:  # raw strategy
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

    return [chunk.strip() for chunk in chunks if chunk.strip()]


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


def to_python_type(obj):
    """
    Recursively converts numpy objects (float32, int64, ndarray, etc.)
    to native Python types for JSON serialization.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(to_python_type(v) for v in obj)
    else:
        return obj
