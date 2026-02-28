FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# On utilise uv pour installer les dépendances de manière synchronisée
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . .

# Exécuter l'orchestrateur via uv
CMD ["uv", "run", "main.py"]