from mcp.server.fastmcp import FastMCP
from core.orchestration.ingestion import fetch_news
from core.orchestration.processor import analyze_data

# Initialisation du serveur
mcp = FastMCP("NewsIntelligence")

@mcp.tool()
async def run_news_pipeline(topic: str):
    """
    Pipeline complet : Ingestion -> Analyse -> Résultat.
    C'est l'outil que l'IA appellera.
    """
    # 1. On récupère
    raw = await fetch_news(topic)
    # 2. On traite
    final = analyze_data(raw)
    # 3. On retourne
    return final