import spacy
from spacy.language import Language

class EntityExtractor:
    """
    Industrial NER (Named Entity Recognition) using spaCy.
    Features: Statistical model + Custom rules to fix common biases.
    """
    def __init__(self, model="en_core_web_sm"):
        print(f"--- Loading NER Model: {model} ---")
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"Downloading model {model}...")
            from spacy.cli import download
            download(model)
            self.nlp = spacy.load(model)
        
        # Add a custom rule to fix the "AI" -> GPE (Geography) error
        self._add_custom_rules()

    def _add_custom_rules(self):
        """
        Adds an EntityRuler to the pipeline to correct statistical errors.
        Example: Ensuring 'AI' is recognized as Technology, not a Country.
        """
        if "entity_ruler" not in self.nlp.pipe_names:
            # Add ruler BEFORE the statistical NER to take priority
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            
            # Define custom patterns
            patterns = [
                {"label": "TECH", "pattern": "AI"},
                {"label": "TECH", "pattern": "Machine Learning"},
                {"label": "PRODUCT", "pattern": [{"LOWER": "m3"}, {"LOWER": "chip"}]},
            ]
            ruler.add_patterns(patterns)

    def extract(self, text):
        """
        Processes text and returns a list of structured entities.
        """
        doc = self.nlp(text)
        results = []
        
        for ent in doc.ents:
            results.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "explanation": spacy.explain(ent.label_)
            })
        return results

    def display_results(self, text):
        """
        Prints entities in a readable format for debugging/CLI.
        """
        entities = self.extract(text)
        print(f"\nOriginal Text: {text}\n")
        print(f"{'ENTITY':<20} | {'LABEL':<10} | {'DESCRIPTION'}")
        print("-" * 60)
        
        for ent in entities:
            print(f"{ent['text']:<20} | {ent['label']:<10} | {ent['explanation']}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Create the extractor
    extractor = EntityExtractor()

    # Sample sentence with technical terms
    sample_text = (
        "Theau launched a local AI project in Paris on February 28, 2026, "
        "using Apple's M3 chip."
    )

    # Run extraction
    extractor.display_results(sample_text)

    # Bonus: Show how it handles multiple entities
    print("\n" + "="*60)
    complex_text = "Elon Musk mentioned that Tesla's AI division is moving to Austin, Texas."
    extractor.display_results(complex_text)