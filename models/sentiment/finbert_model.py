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


class FinBERT:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            Config.BERT_MODEL_NAME
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, encoded_texts: Dict[str, torch.Tensor]) -> List[Dict[str, float]]:
        """Predict sentiment scores for encoded texts."""
        # Move inputs to device
        input_ids = encoded_texts['input_ids'].to(self.device)
        attention_mask = encoded_texts['attention_mask'].to(self.device)

        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Convert logits to probabilities
        probabilities = torch.softmax(outputs.logits, dim=1)

        # Convert to list of dictionaries
        predictions = []
        for probs in probabilities:
            predictions.append({
                'positive': probs[0].item(),
                'negative': probs[1].item(),
                'neutral': probs[2].item()
            })

        return predictions


