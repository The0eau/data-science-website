import numpy as np
from gensim.models import Word2Vec as GensimWord2Vec

class Word2VecScratch:
    """
    A minimalist Skip-Gram Word2Vec implementation using NumPy.
    Demonstrates the internal mechanics of weight matrices and backpropagation.
    """
    def __init__(self, vocab_size, embedding_dim=10, learning_rate=0.01):
        self.v_size = vocab_size
        self.n_dim = embedding_dim
        self.lr = learning_rate
        
        # Initialize weights: 
        # W1: Word embeddings (Input to Hidden)
        # W2: Context weights (Hidden to Output)
        self.W1 = np.random.uniform(-0.5, 0.5, (self.v_size, self.n_dim))
        self.W2 = np.random.uniform(-0.5, 0.5, (self.n_dim, self.v_size))

    def softmax(self, x):
        """Computes softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def train_step(self, target_idx, context_indices):
        """
        Performs one forward and backward pass (Stochastic Gradient Descent).
        """
        # 1. Forward Pass
        h = self.W1[target_idx] # Hidden layer is the embedding of the target word
        u = np.dot(self.W2.T, h) # Scores for all words in vocab
        y_pred = self.softmax(u)

        # 2. Backward Pass (Error calculation)
        e = np.copy(y_pred)
        for idx in context_indices:
            e[idx] -= 1 # Error = Prediction - Ground Truth
        
        # Calculate gradients
        dW2 = np.outer(h, e)
        dW1 = np.dot(self.W2, e)

        # Update weights
        self.W2 -= self.lr * dW2
        self.W1[target_idx] -= self.lr * dW1

    def get_vector(self, word_idx):
        return self.W1[word_idx]


class Word2VecExplorer:
    """
    High-level wrapper for Gensim's Word2Vec implementation.
    Used for production-grade word embeddings.
    """
    def __init__(self, sentences, vector_size=100, window=5):
        self.model = GensimWord2Vec(
            sentences=sentences, 
            vector_size=vector_size, 
            window=window, 
            min_count=1, 
            workers=4
        )

    def get_most_similar(self, word, top_n=3):
        """Returns words with the highest cosine similarity."""
        try:
            return self.model.wv.most_similar(word, topn=top_n)
        except KeyError:
            return f"Word '{word}' not in vocabulary."


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("=== PART 1: Word2Vec FROM SCRATCH (NumPy) ===")
    vocab = {"data": 0, "science": 1, "is": 2, "fun": 3}
    scratch_model = Word2VecScratch(vocab_size=len(vocab), embedding_dim=2)

    print("Training Scratch model on ('data' -> 'science')...")
    for epoch in range(1000):
        scratch_model.train_step(0, [1])

    v_data = scratch_model.get_vector(0)
    v_science = scratch_model.get_vector(1)
    
    # Cosine Similarity calculation
    sim = np.dot(v_data, v_science) / (np.linalg.norm(v_data) * np.linalg.norm(v_science))
    
    print(f"Vector 'data': {v_data}")
    print(f"Vector 'science': {v_science}")
    print(f"Calculated Similarity: {sim:.4f}")

    print("\n=== PART 2: Word2Vec INDUSTRIAL (Gensim) ===")
    corpus = [
        ["king", "rules", "the", "empire"],
        ["queen", "rules", "the", "empire"],
        ["man", "is", "strong"],
        ["woman", "is", "strong"],
        ["apple", "is", "a", "fruit"],
        ["orange", "is", "a", "fruit"]
    ]

    explorer = Word2VecExplorer(corpus, vector_size=10, window=2)
    
    target_word = "king"
    results = explorer.get_most_similar(target_word)
    
    print(f"Most similar words to '{target_word}':")
    for word, score in results:
        print(f" - {word}: {score:.4f}")