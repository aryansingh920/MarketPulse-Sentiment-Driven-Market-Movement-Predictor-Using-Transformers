
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

# tests/test_models.py


class TestModels(unittest.TestCase):
    def setUp(self):
        self.finbert = FinBERT()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.predictor = EnsemblePredictor()

    def test_finbert_model(self):
        """Test FinBERT model functionality."""
        # Create sample input
        encoded_texts = {
            'input_ids': torch.randint(0, 1000, (2, 512)),
            'attention_mask': torch.ones(2, 512)
        }

        # Get predictions
        predictions = self.finbert.predict(encoded_texts)

        # Check predictions structure
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 2)

        for pred in predictions:
            self.assertIn('positive', pred)
            self.assertIn('negative', pred)
            self.assertIn('neutral', pred)

            # Check probabilities sum to approximately 1
            total_prob = sum(pred.values())
            self.assertAlmostEqual(total_prob, 1.0, places=6)

    def test_sentiment_analyzer(self):
        """Test sentiment analyzer functionality."""
        async def run_test():
            # Sample texts
            texts = [
                {
                    'text': 'This stock is performing well',
                    'timestamp': datetime.now()
                },
                {
                    'text': 'Market conditions are concerning',
                    'timestamp': datetime.now()
                }
            ]

            # Analyze texts
            results = await self.sentiment_analyzer.analyze_texts(texts)

            # Check results
            self.assertEqual(len(results), 2)
            for result in results:
                self.assertIn('sentiment_scores', result)
                self.assertIn('sentiment', result)

        asyncio.run(run_test())

    def test_ensemble_predictor(self):
        """Test ensemble predictor functionality."""
        # Create sample data
        prices = pd.DataFrame({
            'close': np.random.randn(100)
        })
        sentiments = [
            {
                'positive': 0.6,
                'negative': 0.2,
                'neutral': 0.2
            }
        ] * 100

        # Test prediction
        predictions = self.predictor.predict(prices, sentiments)

        # Check predictions
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.shape[1], 1)

if __name__ == '__main__':
    unittest.main()
