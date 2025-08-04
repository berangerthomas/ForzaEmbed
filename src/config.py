from dotenv import load_dotenv
from matplotlib.colors import LinearSegmentedColormap

# --- Environment Variables ---
# Load environment variables from .env file
load_dotenv()


# --- Colormap for Heatmaps ---
# This is kept in code as it's a Python object, not easily configured in YAML.
CMAP = LinearSegmentedColormap.from_list(
    "custom",
    ["#2b83ba", "#abdda4", "#ffffbf", "#fdae61", "#d7191c"],
)
