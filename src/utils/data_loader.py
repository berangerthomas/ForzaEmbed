import logging
from pathlib import Path
from typing import List, Tuple, Union


def load_markdown_files(
    data_source: Union[str, Path, List[str]],
) -> List[Tuple[str, str]]:
    """
    Loads markdown content from various sources.

    It can accept:
    1. A directory path (str or Path object) to load all .md files from.
    2. A list of strings, where each string is the markdown content.

    Args:
        data_source: The source of the markdown data.

    Returns:
        A list of tuples, where each tuple contains:
        (name, text).
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
                # Use the filename stem as a generic name
                name = file_path.stem
                all_rows.append((name, content))
    elif isinstance(data_source, list) and all(
        isinstance(item, str) for item in data_source
    ):
        for i, content in enumerate(data_source):
            name = f"Text Content {i + 1}"
            all_rows.append((name, content))
    else:
        raise TypeError(
            "Unsupported data_source type. "
            "Please provide a directory path or a list of markdown strings."
        )
    return all_rows
