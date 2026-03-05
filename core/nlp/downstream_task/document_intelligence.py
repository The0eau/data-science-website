from transformers import pipeline

class DocumentIntelligence:
    """
    Handles Summarization and Question Answering locally.
    """
    def __init__(self):
        print("--- Loading Summarization & Q&A Models ---")
        # Summarization pipeline (BART-based)
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        # Q&A pipeline (BERT-based)
        self.qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    def summarize(self, text, max_length=130, min_length=30):
        """Generates a concise summary of the input text."""
        summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']

    def answer_question(self, context, question):
        """Extracts an answer from a specific context (Extractive QA)."""
        result = self.qa_model(question=question, context=context)
        return result['answer'], result['score']

# --- TEST ---
if __name__ == "__main__":
    engine = DocumentIntelligence()
    
    long_text = """
    Theau is a Data Scientist based in France. He specializes in Natural Language Processing 
    and has built a comprehensive suite of AI tools on his MacBook Air. His project includes 
    vector search with FAISS, entity extraction with spaCy, and local LLM orchestration 
    with LangChain. The goal is to provide a private, high-performance AI experience.
    """
    
    # 1. Test Résumé
    print("\n--- SUMMARY ---")
    print(engine.summarize(long_text))
    
    # 2. Test Q&A
    print("\n--- Q&A ---")
    q = "What is the goal of Theau's project?"
    answer, score = engine.answer_question(long_text, q)
    print(f"Question: {q}")
    print(f"Answer: {answer} (Confidence: {score:.2f})")