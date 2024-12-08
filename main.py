# utils/logger.py
from visualization.dashboard import MarketPulseDashboard
from models.prediction.ensemble_model import EnsemblePredictor
from models.sentiment.sentiment_analyzer import SentimentAnalyzer
from data.collectors.reddit_collector import RedditCollector
from data.collectors.twitter_collector import TwitterCollector
from data.collectors.news_collector import NewsCollector
from utils.database import Database
from utils.logger import Logger
from config.config import Config
from typing import List, Dict
from datetime import datetime, timedelta
import asyncio
from typing import List, Dict, Union
import pandas as pd
import sqlite3
import logging
from pathlib import Path
from datetime import datetime


# main.py


class MarketPulse:
    def __init__(self):
        self.logger = Logger("MarketPulse")
        self.db = Database()

        # Initialize collectors
        self.news_collector = NewsCollector()
        self.twitter_collector = TwitterCollector()
        self.reddit_collector = RedditCollector()

        # Initialize analyzers and predictors
        self.sentiment_analyzer = SentimentAnalyzer()
        self.predictor = EnsemblePredictor()

        # Initialize dashboard
        self.dashboard = MarketPulseDashboard()

    async def collect_data(self, ticker: str, days: int = 7) -> Dict[str, List]:
        """Collect data from all sources."""
        self.logger.info(f"Collecting data for {ticker}")

        # Collect data concurrently
        tasks = [
            self.news_collector.collect_news(ticker, days),
            self.twitter_collector.collect_tweets(ticker, days),
            self.reddit_collector.collect_posts(ticker, days)
        ]

        news_data, twitter_data, reddit_data = await asyncio.gather(*tasks)

        return {
            'news': news_data,
            'twitter': twitter_data,
            'reddit': reddit_data
        }

    async def analyze_sentiment(self, ticker: str, collected_data: Dict[str, List]):
        """Analyze sentiment from collected data."""
        self.logger.info(f"Analyzing sentiment for {ticker}")

        sentiment_results = {}
        for source, data in collected_data.items():
            sentiment_results[source] = await self.sentiment_analyzer.analyze_texts(data)

        # Save results to database
        for source, results in sentiment_results.items():
            self.db.save_sentiment(ticker, results)

        return sentiment_results

    def predict_prices(self, ticker: str, sentiment_data: Dict[str, List]):
        """Make price predictions."""
        self.logger.info(f"Predicting prices for {ticker}")

        # Get historical price data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=Config.LOOKBACK_PERIOD)
        prices = self.db.get_price_data(ticker, start_date, end_date)

        # Aggregate sentiment scores
        aggregated_sentiment = self.sentiment_analyzer.aggregate_sentiment(
            sentiment_data)

        # Make predictions
        predictions = self.predictor.predict(prices, [aggregated_sentiment])

        # Save predictions
        self.db.save_predictions(ticker, predictions)

        return predictions

    def run_dashboard(self):
        """Run the Streamlit dashboard."""
        self.dashboard.run()


async def main():
    app = MarketPulse()

    # Example usage
    ticker = "AAPL"

    # Collect and analyze data
    collected_data = await app.collect_data(ticker)
    sentiment_results = await app.analyze_sentiment(ticker, collected_data)
    predictions = app.predict_prices(ticker, sentiment_results)

    # Run dashboard
    app.run_dashboard()

if __name__ == "__main__":
    asyncio.run(main())
