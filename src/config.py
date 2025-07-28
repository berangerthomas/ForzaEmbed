import os

from dotenv import load_dotenv
from matplotlib.colors import LinearSegmentedColormap

# Import LocalEmbeddingClient from its module
from .local_embedding_client import LocalEmbeddingClient

# --- Configuration ---
# récupérer les variables d'environnement
load_dotenv()
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.6))


# --- Theme and Keyword Generation ---
# Fonction pour générer les thèmes et mots-clés associés aux horaires d'ouverture.
def get_horaires_themes() -> list[str]:
    """
    Retourne une liste de thèmes consolidés pour les horaires.
    """
    return ["horaires", "autre"]


def generate_themes_and_keywords() -> tuple[list[str], dict[str, list[str]]]:
    """
    Génère une liste de thèmes et un dictionnaire de mots-clés pour la détection des horaires d'ouverture.

    Returns:
        tuple: (liste des thèmes, dictionnaire des mots-clés pour les regex)
    """
    lieux = ["bibliothèque", "médiathèque", "mairie", "piscine"]
    jours = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
    actions = ["ouverte", "fermée", "ouverture", "fermeture", "accueil", "horaires"]
    concepts_generaux = [
        "horaires d'ouverture",
        "heures d'ouverture",
        "accueil du public",
        "jour de fermeture",
        "horaires et jours de la semaine",
        "périodes d'ouverture",
        "période scolaire",
        "vacances scolaires",
        "fermeture exceptionnelle",
        "fermeture annuelle",
        "jours fériés",
        "jours spéciaux",
        "horaires spéciaux",
        "quand venir",
        "nous sommes ouverts",
        "nous sommes fermés",
    ]
    expressions_temporelles = [
        "le matin",
        "l'après-midi",
        "de 9h à 12h",
        "de 14h à 18h",
        "de 10h00 à 19h00",
        "9h 12h",
        "14h 18h",
    ]

    # Generate themes by combining actions and days
    themes_jours = [f"{action} le {jour}" for action in actions[:2] for jour in jours]

    # generate themes for each type of place
    themes_lieux = [f"{lieu} {action}" for lieu in lieux for action in actions]

    # Combine all themes into a single list
    base_themes = (
        concepts_generaux + expressions_temporelles + themes_jours + themes_lieux
    )
    base_themes.append("autre")

    # Create a dictionary of keywords for the regex
    keywords = {
        "jours": jours,
        "actions": actions,
    }
    return base_themes, keywords


# --- Grid Search Parameters ---
GRID_SEARCH_PARAMS = {
    "chunk_size": [10, 20, 50, 100, 250],
    "chunk_overlap": [5, 10, 25, 50],
    "themes": {
        "horaires": lambda: generate_themes_and_keywords()[0],
        "horaires_simple": get_horaires_themes,
    },
    "chunking_strategy": ["langchain", "raw"],
}

# --- Model Configuration ---
# MODELS_TO_TEST = [
#     {
#         "type": "local",
#         "name": "paraphrase-multilingual-mpnet-base-v2",
#         "function": LocalEmbeddingClient.get_embeddings,
#     },
#     {
#         "type": "api",
#         "name": "nomic-embed-text",
#         "base_url": "https://api.erasme.homes/v1",
#         "function": lambda client: client.get_embeddings,
#     },
#     {
#         "type": "local",
#         "name": "paraphrase-multilingual-MiniLM-L12-v2",
#         "function": LocalEmbeddingClient.get_embeddings,
#     },
#     {
#         "type": "local",
#         "name": "distiluse-base-multilingual-cased-v1",
#         "function": LocalEmbeddingClient.get_embeddings,
#     },
#     {
#         "type": "local",
#         "name": "distiluse-base-multilingual-cased-v2",
#         "function": LocalEmbeddingClient.get_embeddings,
#     },
# ]
MODELS_TO_TEST = [
    {
        "type": "local",
        "name": "all-mpnet-base-v2",
        "function": LocalEmbeddingClient.get_embeddings,
    },
    # {
    #     "type": "local",
    #     "name": "multi-qa-mpnet-base-dot-v1",
    #     "function": LocalEmbeddingClient.get_embeddings,
    # },
    # {
    #     "type": "local",
    #     "name": "all-distilroberta-v1",
    #     "function": LocalEmbeddingClient.get_embeddings,
    # },
    {
        "type": "local",
        "name": "all-MiniLM-L12-v2",
        "function": LocalEmbeddingClient.get_embeddings,
    },
    # {
    #     "type": "local",
    #     "name": "multi-qa-distilbert-cos-v1",
    #     "function": LocalEmbeddingClient.get_embeddings,
    # },
    # {
    #     "type": "local",
    #     "name": "all-MiniLM-L6-v2",
    #     "function": LocalEmbeddingClient.get_embeddings,
    # },
    # {
    #     "type": "local",
    #     "name": "multi-qa-MiniLM-L6-cos-v1",
    #     "function": LocalEmbeddingClient.get_embeddings,
    # },
    {
        "type": "local",
        "name": "paraphrase-multilingual-mpnet-base-v2",
        "function": LocalEmbeddingClient.get_embeddings,
    },
    # {
    #     "type": "local",
    #     "name": "paraphrase-albert-small-v2",
    #     "function": LocalEmbeddingClient.get_embeddings,
    # },
    {
        "type": "local",
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "function": LocalEmbeddingClient.get_embeddings,
    },
    # {
    #     "type": "local",
    #     "name": "paraphrase-MiniLM-L3-v2",
    #     "function": LocalEmbeddingClient.get_embeddings,
    # },
    {
        "type": "local",
        "name": "distiluse-base-multilingual-cased-v1",
        "function": LocalEmbeddingClient.get_embeddings,
    },
    {
        "type": "local",
        "name": "distiluse-base-multilingual-cased-v2",
        "function": LocalEmbeddingClient.get_embeddings,
    },
    {
        "type": "api",
        "name": "nomic-embed-text",
        "base_url": "https://api.erasme.homes/v1",
        "function": lambda client: client.get_embeddings,
    },
    # {
    #     "type": "api",
    #     "name": "mistral-embed",
    #     "base_url": "https://api.mistral.ai/v1",
    #     "function": lambda client: client.get_embeddings,
    # },
    {
        "type": "api",
        "name": "voyage-3-large",
        "base_url": "https://api.voyageai.com/v1",
        "function": lambda client: client.get_embeddings,
    },
]
# MODELS_TO_TEST = [
#     {
#         "type": "local",
#         "name": "paraphrase-multilingual-mpnet-base-v2",
#         "function": LocalEmbeddingClient.get_embeddings,
#     },
#     {
#         "type": "api",
#         "name": "nomic-embed-text",
#         "base_url": "https://api.erasme.homes/v1",
#         "function": lambda client: client.get_embeddings,
#     },
#     {
#         "type": "api",
#         "name": "mistral-embed",
#         "base_url": "https://api.mistral.ai/v1",
#         "function": lambda client: client.get_embeddings,
#     },
#     {
#         "type": "api",
#         "name": "voyage-3-large",
#         "base_url": "https://api.voyageai.com/v1",
#         "function": lambda client: client.get_embeddings,
#     },
#     {
#         "type": "local",
#         "name": "distiluse-base-multilingual-cased-v1",
#         "function": LocalEmbeddingClient.get_embeddings,
#     },
#     {
#         "type": "local",
#         "name": "distiluse-base-multilingual-cased-v2",
#         "function": LocalEmbeddingClient.get_embeddings,
#     },
# ]
# MODELS_TO_TEST = [
#     {
#         "type": "local",
#         "name": "paraphrase-multilingual-mpnet-base-v2",
#         "function": LocalEmbeddingClient.get_embeddings,
#     },
#     {
#         "type": "api",
#         "name": "nomic-embed-text",
#         "base_url": "https://api.erasme.homes/v1",
#         "function": lambda client: client.get_embeddings,
#     },
# ]
# MODELS_TO_TEST = [
#     {
#         "type": "local",
#         "name": "paraphrase-multilingual-mpnet-base-v2",
#         "function": LocalEmbeddingClient.get_embeddings,
#     },
#     {
#         "type": "local",
#         "name": "paraphrase-multilingual-MiniLM-L12-v2",
#         "function": LocalEmbeddingClient.get_embeddings,
#     },
#     {
#         "type": "local",
#         "name": "distiluse-base-multilingual-cased-v1",
#         "function": LocalEmbeddingClient.get_embeddings,
#     },
#     {
#         "type": "local",
#         "name": "distiluse-base-multilingual-cased-v2",
#         "function": LocalEmbeddingClient.get_embeddings,
#     },
#     {
#         "type": "api",
#         "name": "nomic-embed-text",
#         "base_url": "https://api.erasme.homes/v1",
#         "function": lambda client: client.get_embeddings,
#     },
# ]

# --- Output Configuration ---
# Définit le répertoire de sortie dans le même dossier que le script
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "heatmaps")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Colormap for Heatmaps ---
CMAP = LinearSegmentedColormap.from_list(
    "custom",
    ["#2b83ba", "#abdda4", "#ffffbf", "#fdae61", "#d7191c"],
)
