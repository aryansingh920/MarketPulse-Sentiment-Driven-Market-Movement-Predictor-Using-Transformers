# config/config.py
from pathlib import Path

class Config:
    # Project paths
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / "data"
    MODELS_DIR = ROOT_DIR / "models"
    
    # Data collection settings
    NEWS_SOURCES = [
        "bloomberg.com",
        "reuters.com",
        "finance.yahoo.com"
    ]
    
    SUBREDDITS = [
        "wallstreetbets",
        "stocks",
        "investing"
    ]
    
    # Model parameters
    BERT_MODEL_NAME = "ProsusAI/finbert"
    MAX_LENGTH = 512
    BATCH_SIZE = 32
    
    # Trading parameters
    LOOKBACK_PERIOD = 30  # days
    PREDICTION_HORIZON = 5  # days
    
    # Sentiment analysis
    SENTIMENT_CLASSES = ['positive', 'negative', 'neutral']
    SENTIMENT_WEIGHTS = {
        'news': 0.4,
        'twitter': 0.3,
        'reddit': 0.3
    }

# config/credentials.py
class Credentials:
    # Twitter API credentials
    TWITTER_API_KEY = "your_twitter_api_key"
    TWITTER_API_SECRET = "your_twitter_api_secret"
    TWITTER_ACCESS_TOKEN = "your_twitter_access_token"
    TWITTER_ACCESS_SECRET = "your_twitter_access_secret"
    
    # Reddit API credentials
    REDDIT_CLIENT_ID = "your_reddit_client_id"
    REDDIT_CLIENT_SECRET = "your_reddit_client_secret"
    REDDIT_USER_AGENT = "MarketPulse Bot 1.0"
    
    # Alpha Vantage API
    ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_api_key"
