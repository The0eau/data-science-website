import httpx
import logging

async def fetch_news(topic: str):
    """Récupère des données brutes (Simulation d'API)."""
    url = f"https://api.test.com/news?q={topic}" 
    # Ici on simule une réponse pour que tu puisses tester sans clé API
    return [
        {"title": f"L'IA révolutionne le {topic}", "content": "Contenu détaillé..."},
        {"title": f"Nouveaux modèles de {topic} en 2026", "content": "Analyse..."},
    ]