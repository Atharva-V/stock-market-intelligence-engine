# Implementation Guide & Extension Points

## How to Extend the System

### 1. Adding New Data Sources

**Example: Add Reddit scraping**

```python
# src/scrapers/reddit_scraper.py

import praw
from typing import List
from loguru import logger

class RedditScraper:
    def __init__(self, client_id, client_secret, user_agent):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        logger.add("logs/reddit_scraper.log", rotation="500 MB")
    
    def scrape_subreddits(self, subreddits: List[str], 
                         posts_per_sub: int = 100) -> List[dict]:
        """Scrape posts from multiple subreddits"""
        posts = []
        
        for subreddit_name in subreddits:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            for submission in subreddit.new(limit=posts_per_sub):
                post = {
                    'username': submission.author.name if submission.author else '[deleted]',
                    'timestamp': submission.created_utc,
                    'content': submission.title + ' ' + submission.selftext,
                    'likes': submission.ups,
                    'comments': submission.num_comments,
                    'url': submission.url,
                    'source': 'reddit',
                }
                posts.append(post)
        
        logger.info(f"Scraped {len(posts)} Reddit posts")
        return posts
```

**Integration in main.py:**
```python
# Add to imports
from src.scrapers.reddit_scraper import RedditScraper

# In main():
reddit_scraper = RedditScraper(
    client_id=config.REDDIT_CLIENT_ID,
    client_secret=config.REDDIT_CLIENT_SECRET,
    user_agent=config.REDDIT_USER_AGENT
)
reddit_posts = reddit_scraper.scrape_subreddits(['stocks', 'investing'])
```

---

### 2. Custom Sentiment Models

**Example: ML-based sentiment analysis**

```python
# src/analyzers/ml_sentiment.py

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np

class MLSentimentAnalyzer:
    def __init__(self, model_path=None):
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        self.model = MultinomialNB()
        self.is_trained = False
        
        if model_path:
            self.load_model(model_path)
    
    def train(self, texts: List[str], labels: List[int]):
        """Train on labeled data (0=negative, 1=positive)"""
        tfidf_features = self.tfidf.fit_transform(texts)
        self.model.fit(tfidf_features, labels)
        self.is_trained = True
        logger.info("ML sentiment model trained")
    
    def predict_sentiment(self, text: str) -> float:
        """Predict sentiment [-1, 1]"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        tfidf_features = self.tfidf.transform([text])
        prob = self.model.predict_proba(tfidf_features)[0]
        
        # Convert to [-1, 1]
        sentiment = 2 * prob[1] - 1
        return sentiment
    
    def save_model(self, path: str):
        joblib.dump({
            'tfidf': self.tfidf,
            'model': self.model
        }, path)
    
    def load_model(self, path: str):
        data = joblib.load(path)
        self.tfidf = data['tfidf']
        self.model = data['model']
        self.is_trained = True
```

**Usage in signal_generator.py:**
```python
# Replace SentimentAnalyzer with:
from src.analyzers.ml_sentiment import MLSentimentAnalyzer

# In __init__:
self.sentiment_analyzer = MLSentimentAnalyzer(model_path='models/sentiment.pkl')

# In generate_signals():
signals_df['sentiment_score'] = signals_df['cleaned_content'].apply(
    self.sentiment_analyzer.predict_sentiment
)
```

---

### 3. Custom Visualization

**Example: Interactive Plotly dashboard**

```python
# src/analyzers/interactive_dashboard.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class InteractiveDashboard:
    def __init__(self, output_path: str = "output/dashboards"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def create_dashboard(self, signals_df: pd.DataFrame) -> str:
        """Create interactive Plotly dashboard"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Signal Distribution", "Sentiment Over Time",
                          "Engagement Analysis", "Signal Type Breakdown")
        )
        
        # Signal distribution
        signal_counts = signals_df['signal_type'].value_counts()
        fig.add_trace(
            go.Bar(x=signal_counts.index, y=signal_counts.values, 
                  name='Signals'),
            row=1, col=1
        )
        
        # Sentiment over time
        df_time = signals_df.copy()
        df_time['timestamp'] = pd.to_datetime(df_time['timestamp'])
        
        fig.add_trace(
            go.Scatter(x=df_time['timestamp'], 
                      y=df_time['sentiment_score'],
                      mode='lines+markers',
                      name='Sentiment'),
            row=1, col=2
        )
        
        # Engagement scatter
        fig.add_trace(
            go.Scatter(x=signals_df['signal_confidence'],
                      y=signals_df['total_engagement'],
                      mode='markers',
                      name='Engagement'),
            row=2, col=1
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(labels=signal_counts.index, 
                  values=signal_counts.values,
                  name='Breakdown'),
            row=2, col=2
        )
        
        fig.update_layout(height=1000, showlegend=False)
        
        output_file = self.output_path / "dashboard.html"
        fig.write_html(str(output_file))
        
        logger.info(f"Dashboard saved to {output_file}")
        return str(output_file)
```

---

### 4. Custom Signal Types

**Example: Add "MOMENTUM" signal**

```python
# In signal_generator.py, update _determine_signal_type():

def _determine_signal_type(self, row) -> str:
    """Determine signal type with MOMENTUM"""
    
    if row['signal_confidence'] < self.min_confidence:
        return 'NEUTRAL'
    
    # New: MOMENTUM signals
    if row['engagement_score'] > 0.8 and row['sentiment_score'] > 0.5:
        return 'MOMENTUM_BUY'
    elif row['engagement_score'] > 0.8 and row['sentiment_score'] < -0.5:
        return 'MOMENTUM_SELL'
    
    # Original signals
    elif row['composite_signal'] > 0.3:
        return 'BUY'
    elif row['composite_signal'] < -0.3:
        return 'SELL'
    else:
        return 'HOLD'
```

---

### 5. Real-time Processing

**Example: Stream processing with updates**

```python
# src/stream_processor.py

import time
from datetime import datetime, timedelta
from typing import Callable

class StreamProcessor:
    def __init__(self, check_interval: int = 300):  # 5 minutes
        self.check_interval = check_interval
        self.last_processed_id = None
    
    def process_stream(self, scraper, processor, 
                      signal_gen, callback: Callable):
        """Process tweets continuously"""
        
        logger.info("Starting stream processing")
        
        while True:
            try:
                # Collect new tweets
                tweets = scraper.scrape_search_results(
                    query="#nifty50",
                    max_tweets=100
                )
                
                if tweets:
                    # Process
                    df = processor.process_tweets(tweets)
                    signals_df = signal_gen.generate_signals(df)
                    
                    # Callback (e.g., save to database, send alert)
                    callback(signals_df)
                    
                    logger.info(f"Processed {len(signals_df)} new tweets")
                
                # Wait before next batch
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("Stream processing stopped")
                break
            except Exception as e:
                logger.error(f"Error in stream processing: {e}")
                time.sleep(60)  # Wait before retry
```

---

### 6. Database Integration

**Example: Save to PostgreSQL**

```python
# src/storage/database.py

import psycopg2
from psycopg2.extras import execute_values

class PostgresStorage:
    def __init__(self, connection_string: str):
        self.conn = psycopg2.connect(connection_string)
    
    def save_signals(self, signals_df: pd.DataFrame):
        """Save signals to PostgreSQL"""
        
        cursor = self.conn.cursor()
        
        # Insert signals
        query = """
            INSERT INTO trading_signals 
            (username, timestamp, content, signal_type, 
             composite_signal, confidence, engagement)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        data = [
            (row['username'], row['timestamp'], row['cleaned_content'],
             row['signal_type'], row['composite_signal'], 
             row['signal_confidence'], row['total_engagement'])
            for _, row in signals_df.iterrows()
        ]
        
        execute_values(cursor, query, data)
        self.conn.commit()
        
        logger.info(f"Saved {len(signals_df)} signals to database")
```

---

### 7. API Endpoint

**Example: FastAPI endpoint**

```python
# src/api/app.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class SignalResponse(BaseModel):
    signal_type: str
    strength: float
    confidence: float
    sentiment: float

@app.get("/signals/latest", response_model=List[SignalResponse])
async def get_latest_signals(limit: int = 10):
    """Get latest trading signals"""
    
    df = pd.read_parquet("output/market_data.parquet")
    top_signals = df.nlargest(limit, 'signal_strength')
    
    return [
        SignalResponse(
            signal_type=row['signal_type'],
            strength=float(row['signal_strength']),
            confidence=float(row['signal_confidence']),
            sentiment=float(row['sentiment_score'])
        )
        for _, row in top_signals.iterrows()
    ]

@app.get("/market-summary")
async def get_market_summary():
    """Get market direction and statistics"""
    
    df = pd.read_parquet("output/market_data.parquet")
    
    buy_count = len(df[df['signal_type'] == 'BUY'])
    sell_count = len(df[df['signal_type'] == 'SELL'])
    
    return {
        "market_direction": "BULLISH" if buy_count > sell_count else "BEARISH",
        "buy_signals": int(buy_count),
        "sell_signals": int(sell_count),
        "avg_sentiment": float(df['sentiment_score'].mean())
    }

# Run with: uvicorn src.api.app:app --reload
```

---

### 8. Performance Monitoring

**Example: Metrics collection**

```python
# src/monitoring/metrics.py

from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class ExecutionMetrics:
    start_time: datetime
    end_time: datetime
    tweets_collected: int
    tweets_processed: int
    processing_time_sec: float
    signals_generated: int
    memory_usage_mb: float
    
    def to_dict(self):
        return {
            'timestamp': self.start_time.isoformat(),
            'duration_sec': self.processing_time_sec,
            'tweets_collected': self.tweets_collected,
            'tweets_processed': self.tweets_processed,
            'processing_rate': self.tweets_processed / max(self.processing_time_sec, 1),
            'signals_generated': self.signals_generated,
            'memory_usage_mb': self.memory_usage_mb,
        }
    
    def save(self, path: str = "output/metrics.json"):
        with open(path, 'a') as f:
            json.dump(self.to_dict(), f)
            f.write('\n')
```

---

## Testing Extensions

### Unit Test Example

```python
# tests/test_custom_sentiment.py

import pytest
from src.analyzers.ml_sentiment import MLSentimentAnalyzer

def test_ml_sentiment_training():
    analyzer = MLSentimentAnalyzer()
    
    texts = [
        "I love this stock, it's amazing!",
        "This is terrible, I hate it",
        "Great investment opportunity",
        "Worst decision ever"
    ]
    labels = [1, 0, 1, 0]  # 1=positive, 0=negative
    
    analyzer.train(texts, labels)
    
    assert analyzer.predict_sentiment("I love this") > 0
    assert analyzer.predict_sentiment("I hate this") < 0

def test_ml_sentiment_prediction():
    analyzer = MLSentimentAnalyzer(model_path='models/sentiment.pkl')
    
    score = analyzer.predict_sentiment("#nifty50 bullish breakout")
    assert -1 <= score <= 1
    assert score > 0  # Bullish
```

---

## Configuration for Extensions

Add to `.env`:

```env
# Custom sources
REDDIT_CLIENT_ID=xxx
REDDIT_CLIENT_SECRET=xxx
REDDIT_USER_AGENT=xxx

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=market_intelligence
DB_USER=postgres
DB_PASSWORD=xxx

# API
API_PORT=8000
API_HOST=0.0.0.0

# ML Models
MODEL_PATH=models/sentiment.pkl
MODEL_TRAIN_DATA=data/training.csv

# Real-time
STREAM_CHECK_INTERVAL=300
STREAM_BATCH_SIZE=100
```

---

## Best Practices for Extensions

1. **Follow Existing Patterns**
   - Use same logging setup (loguru)
   - Follow same error handling (try-except-log)
   - Match code style

2. **Add Type Hints**
   ```python
   def process(data: List[str]) -> pd.DataFrame:
       ...
   ```

3. **Document Thoroughly**
   ```python
   def my_function(param1: str) -> dict:
       """
       Brief description.
       
       Args:
           param1: Description of parameter
           
       Returns:
           Description of return value
       """
   ```

4. **Test Before Integration**
   ```python
   # Create test_my_extension.py
   pytest tests/test_my_extension.py -v
   ```

5. **Update Configuration**
   - Add new settings to `.env.example`
   - Update `Config` class
   - Document in README

6. **Monitor Performance**
   - Track execution time
   - Monitor memory usage
   - Log errors comprehensively

---

**Extension Guide Version:** 1.0
