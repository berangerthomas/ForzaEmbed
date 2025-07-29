from pprint import pprint

import pandas as pd
from fastembed import (
    TextEmbedding,
)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

supported_models = (
    pd.DataFrame(TextEmbedding.list_supported_models())
    .sort_values("size_in_GB")
    # .drop(columns=["sources", "model_file", "additional_files"])
    .reset_index(drop=True)
)

pprint(supported_models["model"])

# Enregistre les modèles supportés dans un fichier CSV
supported_models.to_csv("supported_models.csv", index=False)
