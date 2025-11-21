"""
Utility functions and configuration management
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from loguru import logger
import json


class Config:
    """Configuration management"""
    
    def __init__(self, env_file: str = ".env"):
        load_dotenv(env_file)
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment variables"""
        self.SCRAPER_HEADLESS = os.getenv("SCRAPER_HEADLESS", "true").lower() == "true"
        self.SCRAPER_TIMEOUT = int(os.getenv("SCRAPER_TIMEOUT", "30"))
        self.SCRAPER_RATE_LIMIT = float(os.getenv("SCRAPER_RATE_LIMIT", "2.0"))
        
        self.TARGET_TWEET_COUNT = int(os.getenv("TARGET_TWEET_COUNT", "2000"))
        self.SEARCH_KEYWORDS = os.getenv("SEARCH_KEYWORDS", "#nifty50,#sensex,#intraday,#banknifty,#stockmarket,#trading,#stockinvesting").split(",")
        self.TWEET_COLLECTION_HOURS = int(os.getenv("TWEET_COLLECTION_HOURS", "24"))
        
        self.DATA_OUTPUT_PATH = os.getenv("DATA_OUTPUT_PATH", "data/raw")
        self.PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH", "data/processed")
        self.OUTPUT_PARQUET_PATH = os.getenv("OUTPUT_PARQUET_PATH", "output/market_data.parquet")
        
        self.TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", "1000"))
        self.MIN_SIGNAL_CONFIDENCE = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.6"))
        self.VISUALIZATION_SAMPLE_SIZE = int(os.getenv("VISUALIZATION_SAMPLE_SIZE", "500"))
        
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE = os.getenv("LOG_FILE", "logs/market_intelligence.log")
        
        # Create necessary directories
        Path(self.DATA_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
        Path(self.PROCESSED_DATA_PATH).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.OUTPUT_PARQUET_PATH)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.LOG_FILE)).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def setup_logging(log_file: str = "logs/market_intelligence.log", log_level: str = "INFO"):
    """Setup logging configuration"""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logger.remove()  # Remove default handler
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation="500 MB",
        retention="7 days"
    )
    
    logger.add(
        lambda msg: print(msg, end=''),
        format="{time:HH:mm:ss} | {level: <8} | {message}",
        level=log_level
    )
    
    logger.info(f"Logging initialized at {log_file}")
