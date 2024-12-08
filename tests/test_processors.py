
from data.processors.tokenizer import FinanceTokenizer
from data.processors.text_cleaner import TextCleaner
from models.prediction.ensemble_model import EnsemblePredictor
from models.sentiment.sentiment_analyzer import SentimentAnalyzer
from models.sentiment.finbert_model import FinBERT
import numpy as np
import torch
import unittest
import asyncio
from datetime import datetime, timedelta
from data.collectors.news_collector import NewsCollector
from data.collectors.twitter_collector import TwitterCollector
from data.collectors.reddit_collector import RedditCollector

# tests/test_processors.py


class TestProcessors(unittest.TestCase):
    def setUp(self):
        self.cleaner = TextCleaner()
        self.tokenizer = FinanceTokenizer()

    def test_text_cleaner(self):
        """Test text cleaning functionality."""
        # Test text with various elements to clean
        text = "Check out $AAPL stock! http://example.com #investing"
        cleaned = self.cleaner.clean_text(text)

        # Check cleaning results
        self.assertNotIn('$', cleaned)
        self.assertNotIn('http', cleaned)
        self.assertNotIn('#', cleaned)

        # Test stop word removal
        text = "The stock market is volatile today"
        processed = self.cleaner.remove_stopwords(text)
        self.assertNotIn('the', processed.lower())
        self.assertNotIn('is', processed.lower())

    def test_tokenizer(self):
        """Test tokenization functionality."""
        texts = ["Testing the tokenizer", "Another test text"]
        encoded = self.tokenizer.tokenize(texts)

        # Check encoded structure
        self.assertIn('input_ids', encoded)
        self.assertIn('attention_mask', encoded)

        # Check dimensions
        self.assertEqual(len(encoded['input_ids']), 2)
        self.assertEqual(len(encoded['attention_mask']), 2)

if __name__ == '__main__':
    unittest.main()
