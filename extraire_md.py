import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.smart_watch.data_models.schema_bdd import Lieux, ResultatsExtraction

# Remplacez par votre URL de connexion à la base de données
DATABASE_URL = "sqlite:///SmartWatch.db"

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

# Dossier de sortie
output_dir = "exports_markdown"
os.makedirs(output_dir, exist_ok=True)

# Jointure pour récupérer les infos du lieu
resultats = (
    session.query(ResultatsExtraction, Lieux)
    .join(Lieux, ResultatsExtraction.lieu_id == Lieux.identifiant)
    .filter(ResultatsExtraction.markdown_nettoye != None)
    .all()
)

for resultat, lieu in resultats:
    # Nettoyage du nom de fichier
    def safe_filename(s):
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in s or "")

    filename = f"{safe_filename(lieu.identifiant)}_{safe_filename(lieu.type_lieu)}_{safe_filename(lieu.nom)}.md"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(resultat.markdown_nettoye or "")

print(f"{len(resultats)} fichiers exportés dans {output_dir}")
