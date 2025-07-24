import re
from typing import List


# Découpe un texte en phrases ou segments courts.
def chunk_text(text: str) -> List[str]:
    """
    Découpe le texte en segments selon la ponctuation de fin de phrase et les retours à la ligne.

    Args:
        text (str): Texte à découper.

    Returns:
        List[str]: Liste des segments extraits.
    """
    # First, split by sentence-ending punctuation.
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Then, further split each "sentence" by newlines to handle lists
    final_chunks = []
    for sentence in sentences:
        lines = sentence.split("\n")
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:  # Only add non-empty lines
                final_chunks.append(stripped_line)

    return final_chunks


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
def extract_context_around_phrase(
    phrases: list[str], phrase_index: int, context_window: int = 2
) -> str:
    """
    Extrait et met en valeur le contexte autour d'une phrase cible.

    Args:
        phrases (list[str]): Liste des phrases.
        phrase_index (int): Index de la phrase cible.
        context_window (int): Nombre de phrases de contexte à inclure.

    Returns:
        str: Contexte avec la phrase cible mise en évidence.
    """
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
