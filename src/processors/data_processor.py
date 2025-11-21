"""
Data Processing Module
Handles data cleaning, normalization, deduplication, and storage
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Set
from dataclasses import asdict
import re
import unicodedata
from pathlib import Path
import json
from datetime import datetime

from loguru import logger
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field


class Tweet(BaseModel):
    """Pydantic model for tweet validation"""
    username: str
    timestamp: str
    content: str
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    mentions: List[str] = Field(default_factory=list)
    hashtags: List[str] = Field(default_factory=list)
    url: str
    
    class Config:
        arbitrary_types_allowed = True


class DataProcessor:
    """
    Processes, cleans, and normalizes tweet data
    Handles deduplication and storage operations
    """
    
    def __init__(self, output_path: str = "data/processed"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.seen_tweets: Set[str] = set()
        logger.add("logs/processor.log", rotation="500 MB", retention="7 days")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        Handles special characters and Unicode
        """
        try:
            # Handle URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Handle @mentions and #hashtags (preserve them)
            # Remove extra whitespace but preserve mentions and hashtags
            text = re.sub(r'\s+', ' ', text)
            
            # Normalize Unicode
            text = unicodedata.normalize('NFKD', text)
            
            # Remove control characters
            text = ''.join(char for char in text if unicodedata.category(char) != 'Cc')
            
            # Remove emojis (optional, can be preserved for sentiment analysis)
            text = text.encode('ascii', 'ignore').decode('ascii')
            
            return text.strip()
            
        except Exception as e:
            logger.warning(f"Error cleaning text: {e}")
            return text
    
    def normalize_engagement(self, likes: int, retweets: int, replies: int) -> Dict[str, float]:
        """Normalize engagement metrics"""
        total_engagement = likes + retweets + replies
        
        return {
            'likes_norm': likes / max(total_engagement, 1),
            'retweets_norm': retweets / max(total_engagement, 1),
            'replies_norm': replies / max(total_engagement, 1),
            'total_engagement': total_engagement,
            'engagement_score': (likes * 1 + retweets * 2 + replies * 0.5) / max(total_engagement, 1)
        }
    
    def extract_market_signals(self, text: str) -> Dict[str, Any]:
        """Extract market-related signals from tweet text"""
        signals = {
            'mentions_buy_terms': 0,
            'mentions_sell_terms': 0,
            'mentions_bullish_terms': 0,
            'mentions_bearish_terms': 0,
            'sentiment_indicator': 0,  # Will be filled by sentiment analysis
        }
        
        text_lower = text.lower()
        
        buy_terms = ['buy', 'bullish', 'long', 'accumulate', 'green', 'pump', 'moon']
        sell_terms = ['sell', 'bearish', 'short', 'dump', 'red', 'crash', 'bearish']
        bullish_terms = ['bull', 'uptrend', 'resistance', 'breakout', 'surge', 'rally']
        bearish_terms = ['bear', 'downtrend', 'support', 'breakdown', 'drop', 'decline']
        
        signals['mentions_buy_terms'] = sum(1 for term in buy_terms if term in text_lower)
        signals['mentions_sell_terms'] = sum(1 for term in sell_terms if term in text_lower)
        signals['mentions_bullish_terms'] = sum(1 for term in bullish_terms if term in text_lower)
        signals['mentions_bearish_terms'] = sum(1 for term in bearish_terms if term in text_lower)
        
        return signals
    
    def deduplicate_tweets(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate tweets
        Uses content hash to identify duplicates
        """
        unique_tweets = []
        seen_hashes = set()
        
        for tweet in tweets:
            # Create hash from content and username (near-duplicates)
            content_hash = hash((tweet.get('content', ''), tweet.get('username', '')))
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_tweets.append(tweet)
        
        logger.info(f"Deduplicated {len(tweets)} tweets to {len(unique_tweets)} unique tweets")
        return unique_tweets
    
    def process_tweets(self, raw_tweets: List[Any]) -> pd.DataFrame:
        """
        Process raw tweets into structured DataFrame
        
        Args:
            raw_tweets: List of raw tweet data
            
        Returns:
            Processed DataFrame
        """
        processed_data = []
        
        for tweet in raw_tweets:
            try:
                # Clean content
                cleaned_content = self.clean_text(tweet.content if hasattr(tweet, 'content') else tweet.get('content', ''))
                
                # Normalize engagement
                engagement = self.normalize_engagement(
                    tweet.likes if hasattr(tweet, 'likes') else tweet.get('likes', 0),
                    tweet.retweets if hasattr(tweet, 'retweets') else tweet.get('retweets', 0),
                    tweet.replies if hasattr(tweet, 'replies') else tweet.get('replies', 0)
                )
                
                # Extract market signals
                signals = self.extract_market_signals(cleaned_content)
                
                # Parse timestamp
                timestamp_str = tweet.timestamp if hasattr(tweet, 'timestamp') else tweet.get('timestamp', '')
                
                processed_tweet = {
                    'username': tweet.username if hasattr(tweet, 'username') else tweet.get('username', ''),
                    'timestamp': timestamp_str,
                    'original_content': tweet.content if hasattr(tweet, 'content') else tweet.get('content', ''),
                    'cleaned_content': cleaned_content,
                    'likes': engagement['likes_norm'],
                    'retweets': engagement['retweets_norm'],
                    'replies': engagement['replies_norm'],
                    'total_engagement': engagement['total_engagement'],
                    'engagement_score': engagement['engagement_score'],
                    'mentions': tweet.mentions if hasattr(tweet, 'mentions') else tweet.get('mentions', []),
                    'hashtags': tweet.hashtags if hasattr(tweet, 'hashtags') else tweet.get('hashtags', []),
                    'url': tweet.url if hasattr(tweet, 'url') else tweet.get('url', ''),
                    **signals
                }
                
                processed_data.append(processed_tweet)
                
            except Exception as e:
                logger.warning(f"Error processing tweet: {e}")
                continue
        
        df = pd.DataFrame(processed_data)
        
        # Deduplicate
        df = df.drop_duplicates(subset=['username', 'cleaned_content'], keep='first')
        
        # Filter for India relevance
        india_keywords = ["india", "indian", "nse", "bse", "mcx", "mumbai", "delhi", "bangalore", "rupee", "inr"]
        exclude_keywords = ["crypto", "bitcoin", "ethereum", "usa", "us ", "american", "forex"]
        
        def is_india_relevant(text):
            text_lower = text.lower()
            # Exclude non-India content
            if any(keyword in text_lower for keyword in exclude_keywords):
                return False
            # Include if has India-related keywords or market hashtags
            has_india_keyword = any(keyword in text_lower for keyword in india_keywords)
            has_market_tag = any(tag in text_lower for tag in ["nifty", "sensex", "banknifty", "nse", "bse"])
            return has_india_keyword or has_market_tag
        
        # Apply India relevance filter
        initial_count = len(df)
        df = df[df['cleaned_content'].apply(is_india_relevant)]
        filtered_count = initial_count - len(df)
        
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} non-India tweets")
        
        logger.info(f"Processed {len(df)} tweets successfully")
        return df
    
    def save_to_parquet(self, df: pd.DataFrame, filename: str = "market_data.parquet") -> str:
        """
        Save DataFrame to Parquet format for efficient storage
        
        Args:
            df: DataFrame to save
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        try:
            output_file = self.output_path / filename
            
            # Convert to Arrow table
            table = pa.Table.from_pandas(df)
            
            # Write with compression
            pq.write_table(table, str(output_file), compression='snappy')
            
            logger.info(f"Saved {len(df)} records to {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error saving to Parquet: {e}")
            raise
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = "market_data.csv") -> str:
        """Save DataFrame to CSV for accessibility"""
        try:
            output_file = self.output_path / filename
            df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"Saved to {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
            raise
    
    def save_to_json(self, df: pd.DataFrame, filename: str = "market_data.json") -> str:
        """Save DataFrame to JSON"""
        try:
            output_file = self.output_path / filename
            df.to_json(output_file, orient='records', indent=2)
            logger.info(f"Saved to {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
            raise
    
    @staticmethod
    def load_from_parquet(filepath: str) -> pd.DataFrame:
        """Load Parquet file"""
        table = pq.read_table(filepath)
        return table.to_pandas()
    
    def generate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic statistics from processed data"""
        stats = {
            'total_tweets': len(df),
            'unique_users': df['username'].nunique(),
            'date_range': {
                'start': df['timestamp'].min() if len(df) > 0 else None,
                'end': df['timestamp'].max() if len(df) > 0 else None,
            },
            'engagement_stats': {
                'avg_total_engagement': df['total_engagement'].mean(),
                'max_total_engagement': df['total_engagement'].max(),
                'avg_engagement_score': df['engagement_score'].mean(),
            },
            'signal_stats': {
                'avg_buy_mentions': df['mentions_buy_terms'].mean(),
                'avg_sell_mentions': df['mentions_sell_terms'].mean(),
                'avg_bullish_mentions': df['mentions_bullish_terms'].mean(),
                'avg_bearish_mentions': df['mentions_bearish_terms'].mean(),
            },
            'hashtags_frequency': df['hashtags'].explode().value_counts().head(20).to_dict() if len(df) > 0 else {},
        }
        
        return stats
    
    def export_statistics(self, stats: Dict[str, Any], filename: str = "statistics.json") -> str:
        """Export statistics to JSON"""
        try:
            output_file = self.output_path / filename
            
            # Convert numpy types for JSON serialization
            stats_serializable = self._make_serializable(stats)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(stats_serializable, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported statistics to {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error exporting statistics: {e}")
            raise
    
    @staticmethod
    def _make_serializable(obj):
        """Convert numpy/pandas types to JSON-serializable types"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: DataProcessor._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [DataProcessor._make_serializable(item) for item in obj]
        return obj
