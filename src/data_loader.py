import hashlib
import logging
from pathlib import Path
from typing import Any, List, Tuple


def load_markdown_files(
    data_source: Any,
) -> List[Tuple[str, str, str, str]]:
    """
    Loads markdown content from various sources.

    It can accept:
    1. A directory path (str or Path object) to load all .md files from.
    2. A list of strings, where each string is the markdown content.

    Args:
        data_source: The source of the markdown data.

    Returns:
        A list of tuples, where each tuple contains:
        (identifier, name, location_type, text).
    """
    all_rows = []
    if isinstance(data_source, (str, Path)):
        directory = Path(data_source)
        if not directory.is_dir():
            logging.error(f"Data source is not a valid directory: {directory}")
            return []
        for file_path in directory.glob("*.md"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Parsing metadata from filename (e.g., "ID_Type_Libelle.md")
                parts = file_path.stem.split("_", 2)
                identifier = parts[0]
                location_type = parts[1] if len(parts) > 1 else "Unknown"
                name = (
                    parts[2].replace("_", " ") if len(parts) > 2 else "Unknown"
                )  # This is the libellé
                all_rows.append((identifier, name, location_type, content))
    elif isinstance(data_source, list) and all(
        isinstance(item, str) for item in data_source
    ):
        for i, content in enumerate(data_source):
            # Create a unique identifier from the content hash
            identifier = hashlib.sha256(content.encode()).hexdigest()[:16]
            name = f"Text Content {i + 1}"
            location_type = "programmatic"
            all_rows.append((identifier, name, location_type, content))
    else:
        raise TypeError(
            "Unsupported data_source type. "
            "Please provide a directory path or a list of markdown strings."
        )
    return all_rows
