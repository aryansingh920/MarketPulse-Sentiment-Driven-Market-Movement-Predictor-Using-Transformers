# utils/database.py
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


class Database:
    def __init__(self):
        self.db_path = Path("data/market_pulse.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.create_tables()

    def create_tables(self):
        """Create necessary database tables."""
        with self.conn:
            # Sentiment data table
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    sentiment TEXT NOT NULL,
                    positive_score REAL,
                    negative_score REAL,
                    neutral_score REAL,
                    timestamp DATETIME NOT NULL
                )
            ''')

            # Price data table
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS price_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER
                )
            ''')

            # Predictions table
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    predicted_price REAL,
                    confidence_score REAL
                )
            ''')

    def save_sentiment(self, ticker: str, sentiment_data: List[Dict]):
        """Save sentiment analysis results."""
        with self.conn:
            for data in sentiment_data:
                self.conn.execute('''
                    INSERT INTO sentiment_data (
                        ticker, source, text, sentiment,
                        positive_score, negative_score, neutral_score,
                        timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    ticker,
                    data['source'],
                    data['text'],
                    data['sentiment'],
                    data['sentiment_scores']['positive'],
                    data['sentiment_scores']['negative'],
                    data['sentiment_scores']['neutral'],
                    data['timestamp']
                ))

    def save_prices(self, ticker: str, price_data: pd.DataFrame):
        """Save price data."""
        price_data.to_sql(
            'price_data',
            self.conn,
            if_exists='append',
            index=False
        )

    def save_predictions(self, ticker: str, predictions: pd.DataFrame):
        """Save price predictions."""
        predictions.to_sql(
            'predictions',
            self.conn,
            if_exists='append',
            index=False
        )

    def get_sentiment_data(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Retrieve sentiment data for a given period."""
        query = '''
            SELECT *
            FROM sentiment_data
            WHERE ticker = ?
            AND timestamp BETWEEN ? AND ?
        '''
        return pd.read_sql_query(
            query,
            self.conn,
            params=(ticker, start_date, end_date)
        )

    def get_price_data(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Retrieve price data for a given period."""
        query = '''
            SELECT *
            FROM price_data
            WHERE ticker = ?
            AND timestamp BETWEEN ? AND ?
        '''
        return pd.read_sql_query(
            query,
            self.conn,
            params=(ticker, start_date, end_date)
        )
