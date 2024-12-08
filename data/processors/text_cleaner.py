# data/processors/text_cleaner.py
from config.config import Config
from transformers import AutoTokenizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List, Dict, Union


class TextCleaner:
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('punkt')
        self.stop_words = set(stopwords.words('english'))
        # Add financial-specific stop words
        self.stop_words.update(['stock', 'stocks', 'market', 'markets'])

    def clean_text(self, text: str) -> str:
        """Clean and normalize text data."""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove ticker symbols with $ (e.g., $AAPL)
        text = re.sub(r'\$\w*', '', text)

        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def remove_stopwords(self, text: str) -> str:
        """Remove stop words from text."""
        words = word_tokenize(text)
        filtered_words = [
            word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)

    def process_batch(self, texts: List[Dict[str, Union[str, datetime]]]) -> List[str]:
        """Process a batch of text documents."""
        processed_texts = []
        for text_obj in texts:
            if isinstance(text_obj, dict) and 'text' in text_obj:
                cleaned_text = self.clean_text(text_obj['text'])
                cleaned_text = self.remove_stopwords(cleaned_text)
                processed_texts.append(cleaned_text)
            elif isinstance(text_obj, str):
                cleaned_text = self.clean_text(text_obj)
                cleaned_text = self.remove_stopwords(cleaned_text)
                processed_texts.append(cleaned_text)
        return processed_texts


