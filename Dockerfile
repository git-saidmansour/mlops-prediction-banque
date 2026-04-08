# Utilisation d'une image légère et stable
FROM python:3.11-slim

# Définition du répertoire de travail
WORKDIR /app

# Installation des dépendances système nécessaires (si besoin de compiler certains packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie uniquement du fichier requirements pour mettre en cache cette couche Docker
COPY requirements.txt .

# Installation des bibliothèques Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste de l'application
# On s'assure que le dossier mlruns est copié au même niveau que app.py
COPY . .

# Exposition du port utilisé par Flask
EXPOSE 5001

# Variables d'environnement pour Flask et MLflow
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Commande de lancement
CMD ["python", "app.py"]