"""Data loading utilities for ForzaEmbed.

This module provides functions for loading markdown content from various
sources including directories and lists of strings. It handles file I/O
and content extraction.

Example:
    Load markdown files from a directory::

        from src.utils.data_loader import load_markdown_files

        files = load_markdown_files("markdowns/")
        for name, content in files:
            print(f"Loaded: {name}")
"""

import logging
from pathlib import Path
from typing import List, Tuple, Union


def load_markdown_files(
    data_source: Union[str, Path, List[str]],
) -> List[Tuple[str, str]]:
    """Load markdown content from various sources.

    Supports loading from:
    1. A directory path (str or Path) to load all .md files.
    2. A list of strings where each string is markdown content.

    Args:
        data_source: The source of markdown data. Can be a directory path
            or a list of markdown content strings.

    Returns:
        List of tuples containing (name, content) pairs.

    Raises:
        TypeError: If data_source is not a supported type.
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
