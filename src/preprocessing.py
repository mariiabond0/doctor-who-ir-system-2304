import nltk
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt', quiet=True)

def preprocess_text(text: str):
    """Lowercase, remove punctuation, tokenize"""
    if not text:
        return []
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return [t for t in tokens if t.strip()]