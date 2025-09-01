"""
    Little script ot unzip precipitation data from an INMET file.
"""


import zipfile
import os

# ===  Paramètres ===
zip_path = r"/data/home/acoron/LSTM/$2a$10$NMQbLGWAOlncDVIC3UKfuuYuHQIYCI5VTCOksOfEKnK69ySuTaK.zip"  # à adapter si nom différent
extract_dir = r"/data/home/acoron/LSTM/data/precip"

# ===  Création du dossier cible si nécessaire
os.makedirs(extract_dir, exist_ok=True)

# ===  Décompression
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
    print(f"Dézippé : {zip_path} -> {extract_dir}")
