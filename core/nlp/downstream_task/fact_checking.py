from transformers import pipeline

class FactChecker:
    """
    Automated Fact-checking using Natural Language Inference (NLI).
    Verifies if a 'claim' is supported by a 'source' text.
    """
    def __init__(self):
        print("--- Loading Fact-checking Model (RoBERTa-NLI) ---")
        # Lightweight and accurate model for inference
        self.nli_model = pipeline("text-classification", model="roberta-large-mnli")

    def verify(self, context, claim):
        """
        Compares a claim against a context to check its veracity.
        """
        # The model expects the format: "premise [SEP] hypothesis"
        result = self.nli_model(f"{context} </s> {claim}")[0]
        
        label = result['label']
        score = result['score']

        # Mapping RoBERTa labels to human-readable status
        status_map = {
            "CONTRADICTION": "❌ FALSE (Contradicts source)",
            "NEUTRAL": "⚠️ UNVERIFIED (No mention in source)",
            "ENTAILMENT": "✅ TRUE (Supported by source)"
        }

        return {
            "status": status_map.get(label, "UNKNOWN"),
            "confidence": f"{score:.2%}",
            "raw_label": label
        }

# --- TEST ---
if __name__ == "__main__":
    checker = FactChecker()
    
    source_text = "Theau's website was built in 2026 using Python and Apple M3 technology."
    
    claims = [
        "Theau used Python for his website.",              # Entailment
        "Theau's website was built in 1995.",             # Contradiction
        "Theau loves eating Italian pizza in Paris."      # Neutral
    ]

    print(f"\nSource: {source_text}")
    print("-" * 50)
    
    for c in claims:
        res = checker.verify(source_text, c)
        print(f"Claim: {c}")
        print(f"Result: {res['status']} | Confidence: {res['confidence']}\n")