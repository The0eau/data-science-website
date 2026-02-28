class StopwordFilter:
    """
    A tool to remove high-frequency words that carry little semantic meaning.
    
    Attributes:
        stopwords (set): A collection of unique words to be filtered out.
    """

    def __init__(self, language="english"):
        """
        Initializes the filter with a predefined set of stopwords.

        Args:
            language (str): The language for the stopword list ('english' or 'french').
        """
        # Minimalist sets for demonstration
        if language == "english":
            self.stopwords = {
                "a", "an", "the", "and", "but", "if", "or", "because", "as", "what",
                "which", "is", "are", "was", "were", "be", "been", "being",
                "have", "has", "had", "do", "does", "did", "to", "from", "in", "out"
            }
        else: # French
            self.stopwords = {
                "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", 
                "est", "sont", "en", "pour", "dans", "par", "qui", "que", "ne", "pas"
            }

    def filter(self, tokens):
        """
        Removes stopwords from a list of tokens.

        Args:
            tokens (list): A list of strings (tokens) to process.

        Returns:
            list: A new list containing only the informative tokens.
        """
        # We use .lower() to ensure the comparison is case-insensitive
        return [t for t in tokens if t.lower() not in self.stopwords]

# --- MAIN TEST SECTION ---
if __name__ == "__main__":
    # 1. Setup
    english_filter = StopwordFilter(language="english")
    
    # 2. Sample data (tokens from a previous tokenization step)
    raw_tokens = ["The", "data", "science", "is", "an", "amazing", "field", "to", "study"]
    
    # 3. Processing
    clean_tokens = english_filter.filter(raw_tokens)
    
    print("--- STOPWORD FILTERING RESULTS ---")
    print(f"Original: {raw_tokens}")
    print(f"Filtered: {clean_tokens}")
    # Result: ['data', 'science', 'amazing', 'field', 'study']