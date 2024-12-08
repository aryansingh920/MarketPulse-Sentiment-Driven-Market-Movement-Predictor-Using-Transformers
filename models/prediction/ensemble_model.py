# models/prediction/ensemble_model.py
# models/prediction/price_predictor.py
from models.prediction.price_predictor import LSTMPredictor
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict
import torch
import torch.nn as nn


class EnsemblePredictor:
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.price_scaler = MinMaxScaler()
        self.sentiment_scaler = MinMaxScaler()
        self.lookback = Config.LOOKBACK_PERIOD
        self.horizon = Config.PREDICTION_HORIZON

        # Initialize LSTM model
        self.model = LSTMPredictor(
            input_dim=4,  # price + 3 sentiment scores
            hidden_dim=64,
            num_layers=2
        ).to(self.device)

    def prepare_data(
        self,
        prices: pd.DataFrame,
        sentiments: List[Dict[str, float]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for prediction."""
        # Scale price data
        scaled_prices = self.price_scaler.fit_transform(prices[['close']])

        # Convert sentiments to array and scale
        sentiment_array = np.array([[s['positive'], s['negative'], s['neutral']]
                                    for s in sentiments])
        scaled_sentiments = self.sentiment_scaler.fit_transform(
            sentiment_array)

        # Combine features
        features = np.hstack((scaled_prices, scaled_sentiments))

        # Create sequences
        X, y = [], []
        for i in range(len(features) - self.lookback - self.horizon + 1):
            X.append(features[i:(i + self.lookback)])
            y.append(scaled_prices[i + self.lookback:i +
                     self.lookback + self.horizon])

        return (
            torch.FloatTensor(np.array(X)).to(self.device),
            torch.FloatTensor(np.array(y)).to(self.device)
        )

    def train(
        self,
        prices: pd.DataFrame,
        sentiments: List[Dict[str, float]],
        epochs: int = 100
    ):
        """Train the model."""
        X, y = self.prepare_data(prices, sentiments)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def predict(
        self,
        prices: pd.DataFrame,
        sentiments: List[Dict[str, float]]
    ) -> np.ndarray:
        """Make predictions."""
        self.model.eval()

        # Prepare input data
        X, _ = self.prepare_data(
            prices[-self.lookback:], sentiments[-self.lookback:])

        # Make prediction
        prediction = self.model(X)

        # Inverse transform to get actual prices
        return self.price_scaler.inverse_transform(prediction.cpu().numpy())
