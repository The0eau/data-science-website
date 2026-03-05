class EnglishSimpleTokenizer:
    """
    A lightweight tokenizer for English text processing.
    
    This tokenizer handles basic punctuation separation and splits common 
    English contractions (e.g., "I'm", "don't", "John's") into distinct tokens 
    to preserve semantic meaning for downstream NLP tasks.
    """

    def __init__(self):
        """
        Initializes the tokenizer with a default set of punctuation marks.
        """
        self.punctuation = ".,!?;:()[]\""

    def tokenize(self, text):
        """
        Segments a string into a list of tokens.

        Args:
            text (str): The raw input string to be tokenized.

        Returns:
            list: A list of string tokens, with punctuation and contractions 
                  isolated from the main words.
        """
        tokens = []
        current_token = ""
        i = 0
        
        while i < len(text):
            char = text[i]

            # Case 1: Alphanumeric characters
            if char.isalnum():
                current_token += char
            
            # Case 2: English Apostrophe (Handling Contractions & Possession)
            elif char == "'":
                if current_token:
                    tokens.append(current_token)
                    current_token = "'"
                else:
                    current_token = "'"

            # Case 3: Whitespaces
            elif char.isspace():
                if current_token:
                    tokens.append(current_token)
                    current_token = ""

            # Case 4: Standard Punctuation
            elif char in self.punctuation:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append(char)
            
            i += 1

        # Add the final token if it exists
        if current_token:
            tokens.append(current_token)

        return tokens

# --- MAIN TEST SECTION ---
if __name__ == "__main__":
    # Initialize the tokenizer
    tokenizer = EnglishSimpleTokenizer()

    # Test cases: contractions, possession, and nested punctuation
    test_phrases = [
        "I'm learning NLP, it's amazing!",
        "John's car isn't working.",
        "They're going to the U.K. (London)."
    ]

    print("--- ENGLISH TOKENIZER RESULTS ---")
    for i, phrase in enumerate(test_phrases, 1):
        result = tokenizer.tokenize(phrase)
        print(f"\nTest {i}: {phrase}")
        print(f"Tokens: {result}")