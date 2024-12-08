# data/collectors/stock_collector.py
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import asyncio
from config.credentials import Credentials
from config.config import Config
from utils.logger import Logger


class StockCollector:
    def __init__(self):
        self.api_key = Credentials.ALPHA_VANTAGE_API_KEY
        self.base_url = "https://www.alphavantage.co/query"
        self.logger = Logger("StockCollector")

    async def collect_historical_data(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "daily"
    ) -> pd.DataFrame:
        """
        Collect historical stock data from Alpha Vantage.
        
        Args:
            ticker: Stock symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            interval: Data interval ('daily', 'weekly', 'monthly')
            
        Returns:
            DataFrame with historical price data
        """
        try:
            # Construct API parameters
            params = {
                "function": f"TIME_SERIES_{interval.upper()}",
                "symbol": ticker,
                "apikey": self.api_key,
                "outputsize": "full"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        self.logger.error(
                            f"API request failed with status {response.status}")
                        return pd.DataFrame()

                    data = await response.json()

                    # Check for API errors
                    if "Error Message" in data:
                        self.logger.error(
                            f"API returned error: {data['Error Message']}")
                        return pd.DataFrame()

                    # Get the time series data
                    time_series_key = f"Time Series ({interval.capitalize()})"
                    if time_series_key not in data:
                        self.logger.error(
                            f"No time series data found for {ticker}")
                        return pd.DataFrame()

                    # Convert to DataFrame
                    df = pd.DataFrame.from_dict(
                        data[time_series_key], orient='index')

                    # Clean column names
                    df.columns = [col.split('. ')[1].lower()
                                  for col in df.columns]

                    # Convert to numeric types
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col])

                    # Add date column
                    df.index = pd.to_datetime(df.index)
                    df = df.sort_index()

                    # Filter by date range if provided
                    if start_date:
                        df = df[df.index >= start_date]
                    if end_date:
                        df = df[df.index <= end_date]

                    return df

        except Exception as e:
            self.logger.error(f"Error collecting data for {ticker}: {str(e)}")
            return pd.DataFrame()

    async def collect_intraday_data(
        self,
        ticker: str,
        interval: str = "5min"
    ) -> pd.DataFrame:
        """
        Collect intraday stock data from Alpha Vantage.
        
        Args:
            ticker: Stock symbol
            interval: Time interval between data points (1min, 5min, 15min, 30min, 60min)
            
        Returns:
            DataFrame with intraday price data
        """
        try:
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": ticker,
                "interval": interval,
                "apikey": self.api_key,
                "outputsize": "full"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        self.logger.error(
                            f"API request failed with status {response.status}")
                        return pd.DataFrame()

                    data = await response.json()

                    # Check for API errors
                    if "Error Message" in data:
                        self.logger.error(
                            f"API returned error: {data['Error Message']}")
                        return pd.DataFrame()

                    # Get the time series data
                    time_series_key = f"Time Series ({interval})"
                    if time_series_key not in data:
                        self.logger.error(
                            f"No intraday data found for {ticker}")
                        return pd.DataFrame()

                    # Convert to DataFrame
                    df = pd.DataFrame.from_dict(
                        data[time_series_key], orient='index')

                    # Clean column names
                    df.columns = [col.split('. ')[1].lower()
                                  for col in df.columns]

                    # Convert to numeric types
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col])

                    # Add date column
                    df.index = pd.to_datetime(df.index)
                    df = df.sort_index()

                    return df

        except Exception as e:
            self.logger.error(
                f"Error collecting intraday data for {ticker}: {str(e)}")
            return pd.DataFrame()

    async def collect_company_overview(self, ticker: str) -> Dict:
        """
        Collect company overview data from Alpha Vantage.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary with company information
        """
        try:
            params = {
                "function": "OVERVIEW",
                "symbol": ticker,
                "apikey": self.api_key
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        self.logger.error(
                            f"API request failed with status {response.status}")
                        return {}

                    data = await response.json()

                    # Check for API errors
                    if "Error Message" in data:
                        self.logger.error(
                            f"API returned error: {data['Error Message']}")
                        return {}

                    return data

        except Exception as e:
            self.logger.error(
                f"Error collecting company overview for {ticker}: {str(e)}")
            return {}

    async def collect_technical_indicators(
        self,
        ticker: str,
        indicators: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect technical indicators from Alpha Vantage.
        
        Args:
            ticker: Stock symbol
            indicators: List of technical indicators to collect (e.g., ['SMA', 'RSI', 'MACD'])
            
        Returns:
            Dictionary mapping indicator names to DataFrames with indicator data
        """
        results = {}

        try:
            for indicator in indicators:
                params = {
                    "function": indicator,
                    "symbol": ticker,
                    "interval": "daily",
                    "time_period": "14",  # Default time period
                    "series_type": "close",
                    "apikey": self.api_key
                }

                async with aiohttp.ClientSession() as session:
                    async with session.get(self.base_url, params=params) as response:
                        if response.status != 200:
                            self.logger.error(
                                f"API request failed for {indicator} with status {response.status}")
                            continue

                        data = await response.json()

                        # Check for API errors
                        if "Error Message" in data:
                            self.logger.error(
                                f"API returned error for {indicator}: {data['Error Message']}")
                            continue

                        # Get the technical indicator data
                        indicator_key = f"Technical Analysis: {indicator}"
                        if indicator_key not in data:
                            self.logger.error(
                                f"No data found for indicator {indicator}")
                            continue

                        # Convert to DataFrame
                        df = pd.DataFrame.from_dict(
                            data[indicator_key], orient='index')

                        # Convert to numeric types
                        for col in df.columns:
                            df[col] = pd.to_numeric(df[col])

                        # Add date column
                        df.index = pd.to_datetime(df.index)
                        df = df.sort_index()

                        results[indicator] = df

                # Add delay to respect API rate limits
                await asyncio.sleep(0.1)

        except Exception as e:
            self.logger.error(
                f"Error collecting technical indicators for {ticker}: {str(e)}")

        return results

    def calculate_derived_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived technical indicators from price data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicators
        """
        try:
            # Copy input DataFrame
            df = df.copy()

            # Simple Moving Averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['sma_200'] = df['close'].rolling(window=200).mean()

            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_upper'] = df['bb_middle'] + 2 * \
                df['close'].rolling(window=20).std()
            df['bb_lower'] = df['bb_middle'] - 2 * \
                df['close'].rolling(window=20).std()

            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()

            # Volume indicators
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']

            return df

        except Exception as e:
            self.logger.error(
                f"Error calculating derived indicators: {str(e)}")
            return df
