# visualization/plots.py
from models.prediction.ensemble_model import EnsemblePredictor
from models.sentiment.sentiment_analyzer import SentimentAnalyzer
from visualization.plots import MarketPulsePlots
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Union


class MarketPulsePlots:
    @staticmethod
    def create_sentiment_trend(sentiment_data: pd.DataFrame) -> go.Figure:
        """Create sentiment trend visualization."""
        fig = go.Figure()

        # Add traces for each sentiment
        for sentiment in ['positive', 'negative', 'neutral']:
            fig.add_trace(
                go.Scatter(
                    x=sentiment_data.index,
                    y=sentiment_data[sentiment],
                    name=sentiment.capitalize(),
                    mode='lines'
                )
            )

        fig.update_layout(
            title='Sentiment Trends Over Time',
            xaxis_title='Date',
            yaxis_title='Sentiment Score',
            hovermode='x unified',
            legend_title='Sentiment Type'
        )

        return fig

    @staticmethod
    def create_price_prediction(
        historical_prices: pd.DataFrame,
        predicted_prices: pd.DataFrame
    ) -> go.Figure:
        """Create price prediction visualization."""
        fig = go.Figure()

        # Historical prices
        fig.add_trace(
            go.Scatter(
                x=historical_prices.index,
                y=historical_prices['close'],
                name='Historical',
                mode='lines'
            )
        )

        # Predicted prices
        fig.add_trace(
            go.Scatter(
                x=predicted_prices.index,
                y=predicted_prices['predicted'],
                name='Predicted',
                mode='lines',
                line=dict(dash='dash')
            )
        )

        fig.update_layout(
            title='Stock Price: Historical vs Predicted',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified'
        )

        return fig

    @staticmethod
    def create_sentiment_heatmap(sentiment_matrix: pd.DataFrame) -> go.Figure:
        """Create sentiment heatmap visualization."""
        fig = go.Figure(data=go.Heatmap(
            z=sentiment_matrix.values,
            x=sentiment_matrix.columns,
            y=sentiment_matrix.index,
            colorscale='RdYlBu',
            hoverongaps=False
        ))

        fig.update_layout(
            title='Sentiment Analysis Heatmap',
            xaxis_title='Source',
            yaxis_title='Date'
        )

        return fig
