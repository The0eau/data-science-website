#!/bin/bash

# Arrête le script en cas d'erreur
set -e

echo "🚀 Starting Production Orchestrator with uv..."

# Installation/Mise à jour silencieuse des dépendances
uv sync --frozen

# Lancement de Gunicorn avec des workers Uvicorn
# -w 4 : 4 processus esclaves pour la parallélisation
# main:app : cherche l'objet 'app' dans ton fichier 'main.py'
uv run gunicorn main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --log-level info