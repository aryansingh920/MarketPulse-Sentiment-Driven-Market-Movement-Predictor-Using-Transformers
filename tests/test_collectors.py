# tests/test_collectors.py
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


class TestCollectors(unittest.TestCase):
    def setUp(self):
        self.news_collector = NewsCollector()
        self.twitter_collector = TwitterCollector()
        self.reddit_collector = RedditCollector()
        self.ticker = "AAPL"
        self.days = 1

    def test_news_collector(self):
        """Test news collection functionality."""
        async def run_test():
            articles = await self.news_collector.collect_news(self.ticker, self.days)

            # Check if articles were collected
            self.assertIsNotNone(articles)
            self.assertIsInstance(articles, list)

            # Check article structure
            if articles:
                article = articles[0]
                self.assertIn('text', article)
                self.assertIn('timestamp', article)
                self.assertIn('source', article)

        asyncio.run(run_test())

    def test_twitter_collector(self):
        """Test Twitter collection functionality."""
        async def run_test():
            tweets = await self.twitter_collector.collect_tweets(self.ticker, self.days)

            # Check if tweets were collected
            self.assertIsNotNone(tweets)
            self.assertIsInstance(tweets, list)

            # Check tweet structure
            if tweets:
                tweet = tweets[0]
                self.assertIn('text', tweet)
                self.assertIn('created_at', tweet)
                self.assertIn('user', tweet)

        asyncio.run(run_test())

    def test_reddit_collector(self):
        """Test Reddit collection functionality."""
        async def run_test():
            posts = await self.reddit_collector.collect_posts(self.ticker, self.days)

            # Check if posts were collected
            self.assertIsNotNone(posts)
            self.assertIsInstance(posts, list)

            # Check post structure
            if posts:
                post = posts[0]
                self.assertIn('title', post)
                self.assertIn('text', post)
                self.assertIn('created_utc', post)

        asyncio.run(run_test())






if __name__ == '__main__':
    unittest.main()
