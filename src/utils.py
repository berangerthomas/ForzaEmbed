import re
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Découpe un texte en phrases ou segments courts.
def chunk_text(
    text: str, chunk_size: int, chunk_overlap: int, strategy: str = "langchain"
) -> List[str]:
    """
    Découpe le texte en segments de taille et de chevauchement définis.

    Args:
        text (str): Texte à découper.
        chunk_size (int): Taille des chunks.
        chunk_overlap (int): Chevauchement des chunks.
        strategy (str): 'langchain' pour un découpage intelligent, 'raw' pour un découpage brut.

    Returns:
        List[str]: Liste des segments extraits.
    """
    if strategy == "langchain":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
    else:  # raw strategy
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunks.append(text[i : i + chunk_size])

    return [chunk.strip() for chunk in chunks if chunk.strip()]


# Vérifie si un texte contient un motif lié aux horaires d'ouverture.
def contains_horaire_pattern(text: str, keywords: dict) -> bool:
    """
    Vérifie si le texte contient des motifs d'horaires d'ouverture.

    Args:
        text (str): Texte à analyser.
        keywords (dict): Dictionnaire de mots-clés pour la regex.

    Returns:
        bool: True si un motif est trouvé, sinon False.
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


# Extrait le contexte autour d'une phrase cible dans une liste de phrases.
def extract_context_around_phrase(phrases: list[str], phrase_index: int) -> str:
    """
    Extrait et met en valeur le contexte autour d'une phrase cible.

    Args:
        phrases (list[str]): Liste des phrases.
        phrase_index (int): Index de la phrase cible.

    Returns:
        str: Contexte avec la phrase cible mise en évidence.
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
