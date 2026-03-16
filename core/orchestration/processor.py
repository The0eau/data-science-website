def analyze_data(raw_items):
    """Nettoie et enrichit les données."""
    enriched_results = []
    for item in raw_items:
        # Logique simple : on ajoute un score basé sur la longueur du titre
        sentiment_score = len(item['title']) / 100 
        enriched_results.append({
            "headline": item['title'].upper(),
            "relevance": round(sentiment_score, 2),
            "processed": True
        })
    return enriched_results