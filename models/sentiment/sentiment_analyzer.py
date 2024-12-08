# models/sentiment/sentiment_analyzer.py
# models/sentiment/finbert_model.py
from models.sentiment.finbert_model import FinBERT
from data.processors.tokenizer import FinanceTokenizer
from data.processors.text_cleaner import TextCleaner
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Union
import torch
from transformers import AutoModelForSequenceClassification
from typing import List, Dict
from config.config import Config


class SentimentAnalyzer:
    def __init__(self):
        self.cleaner = TextCleaner()
        self.tokenizer = FinanceTokenizer()
        self.model = FinBERT()
        self.weights = Config.SENTIMENT_WEIGHTS

    async def analyze_texts(self, texts: List[Dict[str, Union[str, datetime]]]) -> List[Dict]:
        """Analyze sentiment for a list of texts."""
        # Clean and process texts
        cleaned_texts = self.cleaner.process_batch(texts)

        # Tokenize texts
        encoded_texts = self.tokenizer.tokenize(cleaned_texts)

        # Get sentiment predictions
        predictions = self.model.predict(encoded_texts)

        # Combine original texts with predictions
        results = []
        for text_obj, prediction in zip(texts, predictions):
            if isinstance(text_obj, dict):
                result = {
                    **text_obj,
                    'sentiment_scores': prediction,
                    'sentiment': max(prediction.items(), key=lambda x: x[1])[0]
                }
                results.append(result)

        return results

    def aggregate_sentiment(self, analyzed_data: Dict[str, List[Dict]]) -> Dict[str, float]:
        """Aggregate sentiment scores from multiple sources."""
        source_sentiments = {}

        # Calculate weighted sentiment for each source
        for source, data in analyzed_data.items():
            if not data:
                continue

            # Calculate average sentiment scores for the source
            scores = pd.DataFrame([item['sentiment_scores'] for item in data])
            avg_scores = scores.mean()

            # Apply source weight
            weight = self.weights.get(source, 1.0 / len(analyzed_data))
            source_sentiments[source] = {
                k: v * weight for k, v in avg_scores.items()
            }

        # Combine weighted sentiments
        aggregated = {
            'positive': sum(s['positive'] for s in source_sentiments.values()),
            'negative': sum(s['negative'] for s in source_sentiments.values()),
            'neutral': sum(s['neutral'] for s in source_sentiments.values())
        }

        # Normalize scores
        total = sum(aggregated.values())
        if total > 0:
            aggregated = {k: v/total for k, v in aggregated.items()}

        return aggregated
