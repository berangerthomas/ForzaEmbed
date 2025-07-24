import os


def load_markdown_files(data_dir: str) -> list[tuple[str, str, str, str]]:
    """
    Charge les fichiers Markdown depuis un répertoire.

    Args:
        data_dir (str): Le chemin vers le répertoire contenant les fichiers .md.

    Returns:
        list[tuple[str, str, str, str]]: Une liste de tuples contenant
        (identifiant, nom, type_lieu, contenu).
    """
    markdown_files = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".md"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            # Utilise le nom du fichier sans extension comme identifiant et nom
            file_id = os.path.splitext(filename)[0]
            # Pour 'type_lieu', on peut utiliser une valeur par défaut
            markdown_files.append((file_id, file_id, "markdown", content))
    return markdown_files
