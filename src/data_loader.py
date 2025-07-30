import os


def load_markdown_files(data_dir: str) -> list[tuple[str, str, str, str]]:
    """
    Loads Markdown files from a directory.

    Args:
        data_dir (str): The path to the directory containing .md files.

    Returns:
        list[tuple[str, str, str, str]]: A list of tuples containing
        (identifiant, nom, type_lieu, contenu).
    """
    markdown_files = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".md"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            # Uses the filename without extension as identifiant and nom
            file_id = os.path.splitext(filename)[0]
            # For 'type_lieu', a default value is used
            markdown_files.append((file_id, file_id, "markdown", content))
    return markdown_files
