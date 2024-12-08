# data/collectors/twitter_collector.py
import praw
from datetime import datetime, timedelta
from config.credentials import Credentials
import tweepy
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from config.config import Config


class TwitterCollector:
    def __init__(self):
        auth = tweepy.OAuthHandler(
            Credentials.TWITTER_API_KEY,
            Credentials.TWITTER_API_SECRET
        )
        auth.set_access_token(
            Credentials.TWITTER_ACCESS_TOKEN,
            Credentials.TWITTER_ACCESS_SECRET
        )
        self.api = tweepy.API(auth)

    async def collect_tweets(self, ticker: str, days: int = 7):
        """Collect tweets related to a specific ticker."""
        tweets = []
        query = f"${ticker} -filter:retweets"

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Collect tweets
        async for tweet in tweepy.Cursor(
            self.api.search_tweets,
            q=query,
            lang="en",
            tweet_mode="extended"
        ).items():
            if tweet.created_at < start_date:
                break
            tweets.append({
                'text': tweet.full_text,
                'created_at': tweet.created_at,
                'user': tweet.user.screen_name,
                'likes': tweet.favorite_count,
                'retweets': tweet.retweet_count
            })

        return tweets
