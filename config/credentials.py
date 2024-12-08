# config/credentials.py.example
"""
Example credentials file. Copy this to credentials.py and fill in your API keys.
DO NOT commit credentials.py to version control.
"""


class Credentials:
    # Alpha Vantage API credentials
    # Get your key at: https://www.alphavantage.co/support/#api-key
    ALPHA_VANTAGE_API_KEY = "your_alphavantage_api_key"

    # Twitter API credentials
    # Get your keys at: https://developer.twitter.com/en/portal/dashboard
    TWITTER_API_KEY = "your_twitter_api_key"
    TWITTER_API_SECRET = "your_twitter_api_secret"
    TWITTER_ACCESS_TOKEN = "your_twitter_access_token"
    TWITTER_ACCESS_SECRET = "your_twitter_access_secret"
    TWITTER_BEARER_TOKEN = "your_twitter_bearer_token"

    # Reddit API credentials
    # Get your keys at: https://www.reddit.com/prefs/apps
    REDDIT_CLIENT_ID = "your_reddit_client_id"
    REDDIT_CLIENT_SECRET = "your_reddit_client_secret"
    REDDIT_USERNAME = "your_reddit_username"
    REDDIT_PASSWORD = "your_reddit_password"
    REDDIT_USER_AGENT = "MarketPulse/1.0"

    # Financial News API credentials (optional)
    # Examples for different providers:

    # NewsAPI (https://newsapi.org/)
    NEWSAPI_KEY = "your_newsapi_key"

    # Bloomberg API (if available)
    BLOOMBERG_API_KEY = "your_bloomberg_api_key"
    BLOOMBERG_API_SECRET = "your_bloomberg_api_secret"

    # Reuters API (if available)
    REUTERS_API_KEY = "your_reuters_api_key"
    REUTERS_API_SECRET = "your_reuters_api_secret"

    # Database credentials (if using remote database)
    DB_HOST = "your_database_host"
    DB_PORT = "your_database_port"
    DB_NAME = "your_database_name"
    DB_USER = "your_database_user"
    DB_PASSWORD = "your_database_password"
