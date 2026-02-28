import spacy

def spacy_process_text(text):
    """
    Standard industrial pipeline using SpaCy for full preprocessing.
    
    Args:
        text (str): Input raw text.
        
    Returns:
        list: A list of cleaned, lemmatized tokens.
    """
    # Load the small English model (optimized for speed)
    nlp = spacy.load("en_core_web_sm")
    
    # Process the text: SpaCy performs tokenization, POS tagging, and lemmatization at once
    doc = nlp(text)
    
    # Filtering: keep tokens that are NOT stopwords and NOT punctuation
    return [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]

# --- TEST ---
if __name__ == "__main__":
    raw_input = "The data scientists are analyzing large datasets efficiently!"
    print(f"Industrial Clean: {spacy_process_text(raw_input)}")
    # Output: ['data', 'scientist', 'analyze', 'large', 'dataset', 'efficiently']