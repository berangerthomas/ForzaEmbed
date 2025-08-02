import os

from dotenv import load_dotenv
from matplotlib.colors import LinearSegmentedColormap

from .fastembed_client import FastEmbedClient

# Import embedding clients
from .huggingface_client import get_huggingface_embeddings

# --- Configuration ---
# get environment variables
load_dotenv()
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.6))
SIMILARITY_METRICS = [
    "cosine",
    "euclidean",
    "manhattan",
    "dot_product",
    "chebyshev",
]


# --- Multiprocessing Configuration ---
MULTIPROCESSING_CONFIG = {
    "max_workers_api": 16,  # Maximum workers for API models
    "max_workers_local": None,  # Auto-detect based on CPU count
    "maxtasksperchild": 10,  # Restart workers to prevent memory leaks
    "embedding_batch_size_api": 100,  # Batch size for API embedding requests
    "embedding_batch_size_local": 500,  # Batch size for local embedding requests
    "file_batch_size": 50,  # Batch size for file processing
}


# --- Model Configuration ---
# Define named functions instead of lambdas for multiprocessing compatibility
def api_get_embeddings(client):
    """Named function to replace lambda for API clients."""
    return client.get_embeddings


def fastembed_get_embeddings(client):
    """Named function to replace lambda for FastEmbed clients."""
    return FastEmbedClient.get_embeddings


def huggingface_get_embeddings(client):
    """Named function to replace lambda for HuggingFace clients."""
    return get_huggingface_embeddings


MODELS_TO_TEST = [
    {
        "type": "api",
        "name": "nomic-embed-text",
        "base_url": "https://api.erasme.homes/v1",
        "dimensions": 768,
        "timeout": 240,
        "function": api_get_embeddings,
    },
    {
        "type": "api",
        "name": "mistral-embed",
        "base_url": "https://api.mistral.ai/v1",
        "dimensions": 1024,
        "timeout": 240,
        "function": api_get_embeddings,
    },
    {
        "type": "api",
        "name": "voyage-3-large",
        "base_url": "https://api.voyageai.com/v1",
        "dimensions": 1024,
        "timeout": 240,
        "function": api_get_embeddings,
    },
    {
        "type": "fastembed",
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dimensions": 384,
        "function": FastEmbedClient.get_embeddings,
    },
    {
        "type": "fastembed",
        "name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "dimensions": 768,
        "function": FastEmbedClient.get_embeddings,
    },
    {
        "type": "fastembed",
        "name": "intfloat/multilingual-e5-large",
        "dimensions": 1024,
        "function": FastEmbedClient.get_embeddings,
    },
    {
        "type": "fastembed",
        "name": "jinaai/jina-embeddings-v3",
        "dimensions": 1024,
        "function": FastEmbedClient.get_embeddings,
    },
    {
        "type": "huggingface",
        "name": "Qwen/Qwen3-Embedding-0.6B",
        "dimensions": 1024,
        "function": get_huggingface_embeddings,
    },
    {
        "type": "huggingface",
        "name": "intfloat/multilingual-e5-large-instruct",
        "dimensions": 1024,
        "function": get_huggingface_embeddings,
    },
]

# --- Output Configuration ---
# Sets the output directory in the same folder as the script
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "heatmaps")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Colormap for Heatmaps ---
CMAP = LinearSegmentedColormap.from_list(
    "custom",
    ["#2b83ba", "#abdda4", "#ffffbf", "#fdae61", "#d7191c"],
)

# --- Grid Search Parameters ---
GRID_SEARCH_PARAMS = {
    "chunk_size": [20, 50, 100, 250, 500, 1000],
    # "chunk_size": [100],
    "chunk_overlap": [0, 10, 25, 50, 100, 200],
    # "chunk_overlap": [0],
    "chunking_strategy": ["langchain", "raw", "semchunk", "nltk", "spacy"],
    "similarity_metrics": SIMILARITY_METRICS,
    "themes": {
        "horaires_simple": ["horaires", "autre"],
        "horaires_full": [
            "autre",
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
            "matin",
            "après-midi",
            "soir",
            "matinée",
            "soirée",
            "journée continue",
            "pause déjeuner",
            "pause méridienne",
            "sans interruption",
            "ouverte le lundi",
            "ouverte le mardi",
            "ouverte le mercredi",
            "ouverte le jeudi",
            "ouverte le vendredi",
            "ouverte le samedi",
            "ouverte le dimanche",
            "fermée le lundi",
            "fermée le mardi",
            "fermée le mercredi",
            "fermée le jeudi",
            "fermée le vendredi",
            "fermée le samedi",
            "fermée le dimanche",
            "ouverte le lun",
            "ouverte le mar",
            "ouverte le mer",
            "ouverte le jeu",
            "ouverte le ven",
            "ouverte le sam",
            "ouverte le dim",
            "fermée le lun",
            "fermée le mar",
            "fermée le mer",
            "fermée le jeu",
            "fermée le ven",
            "fermée le sam",
            "fermée le dim",
            "bibliothèque ouverte",
            "bibliothèque fermée",
            "bibliothèque ouverture",
            "bibliothèque fermeture",
            "bibliothèque accueil",
            "bibliothèque horaires",
            "médiathèque ouverte",
            "médiathèque fermée",
            "médiathèque ouverture",
            "médiathèque fermeture",
            "médiathèque accueil",
            "médiathèque horaires",
            "horaires de la bibliothèque",
            "horaires de la médiathèque",
            "heures de consultation",
            "prêt et retour",
            "service public",
            "mairie ouverte",
            "mairie fermée",
            "mairie ouverture",
            "mairie fermeture",
            "mairie accueil",
            "mairie horaires",
            "hôtel de ville",
            "permanence mairie",
            "état civil",
            "services municipaux",
            "accueil mairie",
            "horaires de la mairie",
            "réception du public",
            "piscine ouverte",
            "piscine fermée",
            "piscine ouverture",
            "piscine fermeture",
            "piscine accueil",
            "piscine horaires",
            "bassin ouvert",
            "centre aquatique",
            "horaires de baignade",
            "créneaux libres",
            "nage libre",
            "cours de natation",
            "aquagym",
            "horaires de la piscine",
            "8h00 - 12h00",
            "8h00 - 12h",
            "8h - 12h00",
            "8h - 12h",
            "9h00 - 12h00",
            "9h00 - 12h",
            "9h - 12h00",
            "9h - 12h",
            "10h00 - 12h00",
            "10h00 - 12h",
            "10h - 12h00",
            "10h - 12h",
            "13h00 - 17h00",
            "13h00 - 17h",
            "13h - 17h00",
            "13h - 17h",
            "14h00 - 17h00",
            "14h00 - 17h",
            "14h - 17h00",
            "14h - 17h",
            "13h00 - 18h00",
            "13h00 - 18h",
            "13h - 18h00",
            "13h - 18h",
            "14h00 - 18h00",
            "14h00 - 18h",
            "14h - 18h00",
            "14h - 18h",
            "13h00 - 19h00",
            "13h00 - 19h",
            "13h - 19h00",
            "13h - 19h",
            "14h00 - 19h00",
            "14h00 - 19h",
            "14h - 19h00",
            "14h - 19h",
            "17h00 - 20h00",
            "17h - 20h",
            "18h00 - 21h00",
            "18h - 21h",
            "9h00 - 18h00",
            "9h - 18h",
            "8h30 - 17h30",
            "8h30 - 12h et 14h - 17h30",
            "9h - 12h et 13h30 - 17h",
            "pendant les vacances",
            "hors vacances scolaires",
            "période estivale",
            "saison hivernale",
            "jours ouvrables",
            "jours ouvrés",
            "week-end",
            "samedi matin",
            "dimanche après-midi",
            "actuellement ouvert",
            "actuellement fermé",
            "prochaine ouverture",
            "prochaine fermeture",
            "ouvert aujourd'hui",
            "fermé aujourd'hui",
            "ouvert en continu",
            "fermeture temporaire",
            "de 9h à 12h",
            "de 14h à 17h",
            "entre 9h et 12h",
            "à partir de 9h",
            "jusqu'à 17h",
            "tous les jours sauf",
            "du lundi au vendredi",
            "du mardi au samedi",
            "accès libre",
            "sur rendez-vous",
            "permanence téléphonique",
            "accueil physique",
            "guichet ouvert",
            "service continu",
            "réception ouverte",
            "ouvre ses portes",
            "ferme ses portes",
        ],
    },
}
