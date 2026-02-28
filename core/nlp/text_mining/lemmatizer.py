class Lemmatizer:
    """
    A dictionary-based lemmatizer to normalize words to their root forms.
    
    This implementation uses a lookup table to handle common English 
    inflections and irregular verbs.
    """

    def __init__(self):
        """
        Initializes the lemmatizer with a mapping of inflected forms to lemmas.
        """
        # In a real scenario, this would be a file with 100,000+ entries
        self.lemma_lookup = {
            "running": "run",
            "ran": "run",
            "runs": "run",
            "ate": "eat",
            "eaten": "eat",
            "eating": "eat",
            "better": "good",
            "worst": "bad",
            "rocks": "rock",
            "corpora": "corpus"
        }

    def lemmatize(self, token):
        """
        Returns the lemma of a given token.

        Args:
            token (str): The word to be lemmatized.

        Returns:
            str: The base form (lemma) if found, otherwise the original token.
        """
        word = token.lower()
        return self.lemma_lookup.get(word, word)

# --- MAIN TEST SECTION ---
if __name__ == "__main__":
    lemmatizer = Lemmatizer()
    
    # Test tokens
    tokens = ["The", "runner", "ran", "better", "than", "the", "worst", "competitor"]
    
    # Simple lemmatization (note: 'runner' is not in our small dictionary)
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    
    print("--- LEMMATIZATION RESULTS ---")
    print(f"Tokens: {tokens}")
    print(f"Lemmas: {lemmas}")
    # Result: ['the', 'runner', 'run', 'good', 'than', 'the', 'bad', 'competitor']