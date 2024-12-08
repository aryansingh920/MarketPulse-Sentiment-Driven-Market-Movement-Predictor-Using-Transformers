# data/collectors/reddit_collector.py
import praw
from datetime import datetime, timedelta
from config.credentials import Credentials
import tweepy
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from config.config import Config


class RedditCollector:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=Credentials.REDDIT_CLIENT_ID,
            client_secret=Credentials.REDDIT_CLIENT_SECRET,
            user_agent=Credentials.REDDIT_USER_AGENT
        )
        self.subreddits = Config.SUBREDDITS

    async def collect_posts(self, ticker: str, days: int = 7):
        """Collect Reddit posts and comments related to a specific ticker."""
        posts = []

        for subreddit_name in self.subreddits:
            subreddit = self.reddit.subreddit(subreddit_name)

            # Search for posts containing the ticker
            async for post in subreddit.search(f"${ticker}", time_filter="week"):
                posts.append({
                    'title': post.title,
                    'text': post.selftext,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'subreddit': subreddit_name
                })

        return posts
