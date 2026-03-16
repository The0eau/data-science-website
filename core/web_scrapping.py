import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def fetch_arxiv_papers(query, max_results=5):
    """
    EXTRACT: Utilisation de l'API d'arXiv pour récupérer des métadonnées.
    """
    print(f"--- Step 1: Querying arXiv API for '{query}' ---")
    base_url = "http://export.arxiv.org/api/query?"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results
    }
    
    response = requests.get(base_url, params=params)
    soup = BeautifulSoup(response.content, 'xml') # XML Parsing pour l'API
    
    papers = []
    for entry in soup.find_all('entry'):
        paper_data = {
            'title': entry.title.text.strip().replace('\n', ''),
            'published': entry.published.text,
            'link': entry.id.text,
            'summary': entry.summary.text.strip()
        }
        papers.append(paper_data)
    
    return papers

def simulate_scraping_details(papers):
    """
    TRANSFORM: Simule le passage sur chaque page pour un "deep scraping".
    """
    print("--- Step 2: Simulating deep scraping for keywords ---")
    for paper in papers:
        # Ici, on pourrait faire un requests.get(paper['link'])
        # Pour l'exemple, on transforme le résumé en liste de tags (Parsing textuel)
        text = paper['summary'].lower()
        potential_keywords = ['neural', 'learning', 'transformer', 'data', 'model']
        found_tags = [tag for tag in potential_keywords if tag in text]
        
        paper['tags'] = ", ".join(found_tags) if found_tags else "General AI"
        time.sleep(1) # Bonne pratique : on ne bombarde pas le serveur
        
    return papers

# --- EXÉCUTION DU PIPELINE ---
# 1. Extraction
raw_papers = fetch_arxiv_papers("artificial intelligence")

# 2. Transformation / Scraping enrichi
enriched_papers = simulate_scraping_details(raw_papers)

# 3. Chargement (Load) en DataFrame et CSV
df = pd.DataFrame(enriched_papers)
df.to_csv("ai_research_dataset.csv", index=False)

print("\n--- Pipeline Complete: 'ai_research_dataset.csv' created ---")
print(df[['title', 'tags']].head())