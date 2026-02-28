import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class VectorSearchEngine:
    """
    Industrial Retrieval system using FAISS and Sentence Transformers.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def build_index(self, documents):
        """
        Encodes documents and adds them to a FAISS index.
        """
        self.documents = documents
        # 1. Convert text to embeddings
        embeddings = self.model.encode(documents)
        
        # 2. Initialize FAISS index (IndexFlatL2 uses Euclidean distance)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        
        # 3. Add vectors to the index
        self.index.add(np.array(embeddings).astype('float32'))
        print(f"Index built with {self.index.ntotal} documents.")

    def search(self, query, top_k=2):
        """
        Finds the most semantically similar documents to the query.
        """
        # 1. Encode the query
        query_vector = self.model.encode([query]).astype('float32')
        
        # 2. Search in FAISS
        distances, indices = self.index.search(query_vector, top_k)
        
        # 3. Format results
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            results.append({
                "text": self.documents[idx],
                "score": float(distances[0][i])
            })
        return results

# --- TEST ---
if __name__ == "__main__":
    docs = [
        "The Eiffel Tower is located in Paris, France.",
        "Python is a versatile programming language for Data Science.",
        "The Great Wall of China is a historic landmark.",
        "Machine Learning models require high-quality data."
    ]

    engine = VectorSearchEngine()
    engine.build_index(docs)

    query = "Tell me about French monuments"
    matches = engine.search(query, top_k=1)

    print(f"\nQuery: {query}")
    print(f"Top Result: {matches[0]['text']} (Distance: {matches[0]['score']:.4f})")