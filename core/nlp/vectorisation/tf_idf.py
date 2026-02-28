import math

class SimpleTFIDF:
    """
    A basic TF-IDF calculator to understand word importance in a corpus.
    """

    def __init__(self, corpus):
        """
        Args:
            corpus (list of list): List of tokenized documents.
        """
        self.corpus = corpus
        self.num_docs = len(corpus)
        # Pre-calculate how many documents contain each word
        self.idf_counts = self._calculate_idf_counts()

    def _calculate_idf_counts(self):
        counts = {}
        for doc in self.corpus:
            # We use a set to count each unique word only once per document
            for word in set(doc):
                counts[word] = counts.get(word, 0) + 1
        return counts

    def get_tfidf(self, word, doc):
        """
        Calculates the TF-IDF score for a specific word in a specific document.
        """
        # 1. Calculate TF
        tf = doc.count(word) / len(doc)
        
        # 2. Calculate IDF
        docs_with_word = self.idf_counts.get(word, 0)
        # We add 1 to avoid division by zero
        idf = math.log(self.num_docs / (1 + docs_with_word))
        
        return tf * idf

# --- MAIN TEST ---
if __name__ == "__main__":
    # Example corpus (tokenized)
    corpus = [
        ["data", "science", "is", "awesome"],
        ["data", "science", "is", "about", "data"],
        ["the", "weather", "is", "sunny"]
    ]
    
    tfidf_tool = SimpleTFIDF(corpus)
    
    # Word 'data' in doc 1 (Common word)
    score_data = tfidf_tool.get_tfidf("data", corpus[1])
    # Word 'sunny' in doc 2 (Unique word)
    score_sunny = tfidf_tool.get_tfidf("sunny", corpus[2])
    
    print(f"TF-IDF of 'data': {score_data:.4f}")
    print(f"TF-IDF of 'sunny': {score_sunny:.4f}")