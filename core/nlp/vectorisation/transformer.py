import numpy as np
from sentence_transformers import SentenceTransformer, util

# ==========================================
# PART 1: TRANSFORMER ENCODER BLOCK FROM SCRATCH
# (The "Brain" of modern NLP)
# ==========================================

class MiniTransformerBlock:
    """
    A minimalist Transformer Encoder block using NumPy.
    Demonstrates Self-Attention: how words 'pay attention' to each other.
    """
    def __init__(self, d_model=4):
        self.d_model = d_model
        # Weights for Query, Key, and Value
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)
        # Feed-Forward weight
        self.W_ff = np.random.randn(d_model, d_model)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def self_attention(self, x):
        # Linear projections
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)

        # Scaled Dot-Product Attention
        # scores = (Q * K^T) / sqrt(d_k)
        scores = np.dot(Q, K.T) / np.sqrt(self.d_model)
        weights = self.softmax(scores)
        
        # Output is the weighted sum of Values
        return np.dot(weights, V), weights

    def forward(self, x):
        # 1. Attention mechanism
        attn_out, weights = self.self_attention(x)
        # 2. Add (Residual connection)
        x = x + attn_out
        # 3. Feed Forward layer
        ff_out = np.dot(x, self.W_ff)
        # 4. Add (Residual connection)
        return x + ff_out, weights



# ==========================================
# PART 2: INDUSTRIAL SEMANTIC SEARCH
# (State-of-the-art SBERT)
# ==========================================

class IndustrialSemanticSearch:
    """
    High-level implementation using SBERT (Sentence-BERT).
    Captures deep contextual meaning between sentences.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # This pre-trained model understands English nuances perfectly
        self.model = SentenceTransformer(model_name)

    def get_similarity(self, sent1, sent2):
        # Transform sentences into 384-dimensional vectors
        embeddings = self.model.encode([sent1, sent2])
        # Compute Cosine Similarity between the two vectors
        return util.cos_sim(embeddings[0], embeddings[1]).item()



# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("=== 1. TEST: TRANSFORMER BLOCK FROM SCRATCH ===")
    # Input: 3 words (e.g., "Data is gold"), each a 4D vector
    input_vectors = np.array([
        [1, 0, 0, 0], # Word 1
        [0, 1, 0, 0], # Word 2
        [0, 0, 1, 0]  # Word 3
    ])

    transformer = MiniTransformerBlock(d_model=4)
    output, attn_matrix = transformer.forward(input_vectors)

    print(f"Output Matrix shape: {output.shape}")
    print("Attention Matrix (Who looks at who?):\n", attn_matrix.round(2))
    print("\n" + "="*50 + "\n")

    print("=== 2. TEST: INDUSTRIAL SEMANTIC SEARCH ===")
    s1 = "A man is playing guitar."
    s2 = "A person is strumming a musical instrument."
    s3 = "The weather is quite cold today."

    search_engine = IndustrialSemanticSearch()
    
    sim_high = search_engine.get_similarity(s1, s2)
    sim_low = search_engine.get_similarity(s1, s3)
    
    print(f"Sentence A: '{s1}'")
    print(f"Sentence B: '{s2}'")
    print(f"Similarity A vs B: {sim_high:.4f} (Context understood!)")
    
    print(f"\nSentence C: '{s3}'")
    print(f"Similarity A vs C: {sim_low:.4f} (Different topics)")