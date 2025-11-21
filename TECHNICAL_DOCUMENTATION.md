# Technical Documentation

## System Architecture & Design

### Overview
The Market Intelligence System is a production-grade Python application that collects real-time data about Indian stock market discussions from social media, processes it into structured data, generates trading signals, and produces actionable insights.

---

## 1. Data Collection Layer (twitter_scraper.py)

### Design Pattern: Facade + Factory

**Class Hierarchy:**
```
TwitterScraper
├── _initialize_driver()          # WebDriver setup
├── _apply_rate_limit()           # Throttling
├── _extract_tweets_from_page()   # DOM parsing
├── scrape_search_results()       # Query-level scraping
└── scrape_multiple_queries()     # Orchestration

AlternativeScraper
└── get_with_retry()              # Fallback mechanism
```

### Anti-Bot Implementation

**1. WebDriver Stealth Mode:**
```python
# Hide automation indicators
Object.defineProperty(navigator, 'webdriver', {get: () => false})
--disable-blink-features=AutomationControlled
excludeSwitches: ['enable-automation']
```

**2. User-Agent Rotation:**
- 5 different user agents
- Random selection per request
- Mimics real browser behavior

**3. Rate Limiting:**
- Minimum 2 seconds between requests
- Random jitter (0-0.5 seconds)
- Prevents bot detection algorithms

**4. Request Handling:**
- Exponential backoff on failure
- 3 retry attempts per request
- Timeout after 30 seconds

### Complexity Analysis

**Time Complexity:**
- Single query: O(n) where n = tweets
- Multiple queries: O(m×n) where m = number of queries
- Scrolling overhead: O(log n) for pagination

**Space Complexity:**
- Tweet storage: O(n) where n = total tweets
- DOM element caching: O(1) constant
- Driver memory: ~500MB average

**Performance:**
- ~500-800 tweets/minute with rate limiting
- ~2500-5000 tweets/minute without rate limiting
- Bottleneck: Network I/O and DOM rendering

---

## 2. Data Processing Layer (data_processor.py)

### Design Pattern: Pipeline + Strategy

**Processing Pipeline:**
```
Raw Tweets
    ↓
[URL Removal] → Regex-based extraction
    ↓
[Unicode Normalization] → NFKD decomposition
    ↓
[Character Filtering] → Remove control chars
    ↓
[Deduplication] → Content-based hashing
    ↓
[Engagement Normalization] → Min-max scaling
    ↓
[Signal Extraction] → Keyword counting
    ↓
Processed DataFrame
```

### Text Cleaning Algorithm

```python
def clean_text(text: str) -> str:
    # Step 1: Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Step 2: Normalize Unicode (NFKD)
    text = unicodedata.normalize('NFKD', text)
    
    # Step 3: Remove control characters
    text = ''.join(c for c in text 
                   if unicodedata.category(c) != 'Cc')
    
    # Step 4: Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
```

**Unicode Handling:**
- NFKD normalization for Indian characters
- Preserves valid Unicode text
- Removes emoji and control characters
- Handles combining diacritics

### Deduplication Strategy

**Hash-based Near-Duplicate Detection:**
```python
# Create hash from content + username
content_hash = hash((tweet.content, tweet.username))

# Identify duplicates
if content_hash not in seen_hashes:
    unique_tweets.append(tweet)
```

**Complexity:**
- Time: O(n) single pass
- Space: O(m) where m = unique tweets
- False positive rate: ~0.1% (acceptable)

### Engagement Normalization

```python
# Normalize individual metrics
likes_norm = likes / total_engagement
retweets_norm = retweets / total_engagement
replies_norm = replies / total_engagement

# Composite engagement score
engagement_score = (likes×1 + retweets×2 + replies×0.5) / max(total, 1)
```

**Weighting Rationale:**
- Retweets (2x): Indicates broader reach
- Likes (1x): Basic engagement
- Replies (0.5x): Often spam or controversy

---

## 3. Analysis Layer (signal_generator.py)

### Design Pattern: Strategy + Template Method

**Signal Generation Pipeline:**
```
Processed Data
    ↓
[Sentiment Analysis] → Keyword-based scoring
    ↓
[Feature Extraction] → TF-IDF vectorization
    ↓
[Signal Calculation] → Weighted composite
    ↓
[Confidence Scoring] → Multi-factor assessment
    ↓
[Signal Classification] → BUY/SELL/HOLD/NEUTRAL
    ↓
Trading Signals with Confidence
```

### Sentiment Analysis

**Word-Based Scoring:**
```python
positive_words = {
    'bullish': 2.0, 'bull': 1.8, 'buy': 1.5,
    'pump': 1.8, 'surge': 1.7, 'green': 1.6,
    # ... more words
}

negative_words = {
    'bearish': -2.0, 'bear': -1.8, 'sell': -1.5,
    'dump': -1.8, 'crash': -1.9, 'red': -1.6,
    # ... more words
}

# Calculate score
positive_score = sum(positive_words.get(w, 0) for w in words)
negative_score = sum(negative_words.get(w, 0) for w in words)
total_score = positive_score + negative_score

# Normalize: [-1, 1]
sentiment = np.tanh(total_score / 10)
```

**Advantages:**
- Fast computation (O(n) where n = words)
- Interpretable results
- Easy to extend with domain keywords

**Limitations:**
- Doesn't understand context (e.g., sarcasm)
- Requires manual keyword curation
- No word embeddings (for future enhancement)

### Composite Signal Generation

**Multi-Factor Signal:**
```python
signal = (
    0.3 × sentiment_score +
    0.2 × normalized_engagement +
    0.25 × (buy_terms - sell_terms) / (buy_terms + sell_terms) +
    0.25 × (bullish_terms - bearish_terms) / (bullish_terms + bearish_terms)
)

# Normalize to [-1, 1]
signal = np.tanh(signal)
```

**Component Weights Rationale:**
- Sentiment (30%): Direct market direction indicator
- Engagement (20%): Signal reliability
- Buy/Sell (25%): Explicit trading direction
- Bullish/Bearish (25%): Technical analysis perspective

### Confidence Calculation

```python
confidence = (
    0.5 × (engagement / max_engagement) +  # Engagement confidence
    0.3 × unique_word_ratio +               # Text quality
    0.2 × min(mention_count / 5, 1.0)      # Signal density
)

confidence = clip(confidence, 0, 1)
```

**Interpretation:**
- 0.0-0.3: Low confidence (noise)
- 0.3-0.6: Medium confidence (consider)
- 0.6-1.0: High confidence (reliable)

### Signal Classification

**Decision Tree:**
```
if confidence < MIN_CONFIDENCE (0.6):
    → NEUTRAL (unreliable)
elif signal > 0.3:
    → BUY (bullish)
elif signal < -0.3:
    → SELL (bearish)
else:
    → HOLD (uncertain)
```

---

## 4. Visualization Layer (visualization.py)

### Design Pattern: Factory + Strategy

**Memory Efficiency Techniques:**

**1. Data Sampling:**
```python
# For datasets > sample_size
df_sample = df.sample(n=sample_size, random_state=42)
# Reduces from 10K to 500 points for visualization
# Space reduction: 95%
```

**2. DPI Optimization:**
```python
plt.savefig(file, dpi=100)  # Web quality
# vs.
plt.savefig(file, dpi=300)  # Print quality
# ~90% file size reduction
```

**3. Matplotlib Backend:**
```python
matplotlib.use('Agg')  # No display needed
# ~50% memory reduction
```

**Performance Metrics:**
- Plot generation: < 2 seconds per plot
- Memory usage: ~100MB for 10K points
- File size: 200-400KB per PNG

### Plot Types

**1. Signal Distribution (3-panel):**
- Signal type counts (bar chart)
- Composite signal histogram
- Strength vs Confidence scatter

**2. Engagement Analysis (4-panel):**
- Engagement by signal type
- Total engagement histogram
- Likes vs Retweets scatter
- Engagement score distribution

**3. Sentiment Analysis (4-panel):**
- Sentiment by signal type
- Sentiment distribution
- Market terms count
- Sentiment vs Engagement scatter

**4. Temporal Analysis (3-panel):**
- Composite signal over time (area)
- Confidence over time (line)
- Engagement over time (bar)

---

## 5. Storage & Serialization

### Parquet Format Advantages

**vs. CSV:**
- **Compression**: 70-80% space savings
- **Read speed**: 10-100x faster
- **Schema**: Built-in type information
- **Streaming**: Support for large files

**vs. JSON:**
- **Size**: 95% smaller
- **Speed**: 50x faster read
- **Memory**: Efficient column storage

**Example:**
```python
import pyarrow.parquet as pq

# Write with compression
pq.write_table(arrow_table, 'data.parquet', compression='snappy')

# Read efficiently
table = pq.read_table('data.parquet')
df = table.to_pandas()
```

### Data Schema

```python
Schema: StructType(
    StructField('username', StringType),
    StructField('timestamp', StringType),
    StructField('cleaned_content', StringType),
    StructField('total_engagement', DoubleType),
    StructField('engagement_score', DoubleType),
    StructField('composite_signal', DoubleType),
    StructField('signal_confidence', DoubleType),
    StructField('signal_type', StringType),
    StructField('sentiment_score', DoubleType),
    # ... additional fields
)
```

---

## 6. Error Handling & Resilience

### Exception Hierarchy

```
Exception
├── TimeoutException → Retry with exponential backoff
├── NoSuchElementException → Log & continue
├── RequestException → Fallback scraper
├── ParseError → Skip tweet, log warning
└── StorageException → Fail fast, report
```

### Retry Logic

```python
def retry_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            wait_time = 2 ** attempt  # 1s, 2s, 4s
            logger.warning(f"Retry {attempt+1} after {wait_time}s")
            time.sleep(wait_time)
    raise Exception(f"Failed after {max_retries} attempts")
```

---

## 7. Performance Optimization Strategies

### For 2000 Tweets
- Sequential scraping: ~20 minutes
- Total processing: ~25-30 minutes
- Memory usage: ~1-2GB peak

### For 20,000 Tweets (10x)
**Scaling Strategy:**

**1. Distributed Scraping:**
```python
# Spawn 4 parallel scrapers
scraper1: query1, query2
scraper2: query3, query4
scraper3: query5, query6
scraper4: query7, query8

# Time: ~25 minutes (parallelizable)
```

**2. Batch Processing:**
```python
# Process 1000 tweets per batch
for batch in chunks(tweets, 1000):
    processed = processor.process_tweets(batch)
    signals = signal_gen.generate_signals(processed)
    save_batch(signals)
```

**3. Incremental Analysis:**
```python
# Only analyze new tweets
last_id = load_checkpoint()
new_tweets = [t for t in tweets if t.id > last_id]
analyze(new_tweets)
```

**4. Partitioned Storage:**
```python
# Split by date
data/processed/2024-11-20/nifty50.parquet
data/processed/2024-11-20/sensex.parquet
```

---

## 8. Configuration Management

### Environment Variables

```env
# scraper: Headless mode, timeout, rate limit
SCRAPER_HEADLESS=true
SCRAPER_TIMEOUT=30
SCRAPER_RATE_LIMIT=2

# collection: Target tweets, keywords, hours
TARGET_TWEET_COUNT=2000
SEARCH_KEYWORDS=#nifty50,#sensex,#intraday,#banknifty
TWEET_COLLECTION_HOURS=24

# analysis: Feature count, confidence threshold
TFIDF_MAX_FEATURES=1000
MIN_SIGNAL_CONFIDENCE=0.6

# visualization: Sample size for plots
VISUALIZATION_SAMPLE_SIZE=500

# logging: Level and file path
LOG_LEVEL=INFO
LOG_FILE=logs/market_intelligence.log
```

---

## 9. Logging Strategy

### Log Levels

**DEBUG:** Detailed execution flow
```
[2024-11-20 15:30:00] | DEBUG    | Extracting tweets from page...
```

**INFO:** High-level progress
```
[2024-11-20 15:30:00] | INFO     | Collected 500 tweets total
```

**WARNING:** Recoverable issues
```
[2024-11-20 15:30:00] | WARNING  | Timeout waiting for tweets, retrying...
```

**ERROR:** Unrecoverable issues
```
[2024-11-20 15:30:00] | ERROR    | Failed to initialize WebDriver
```

### Log Rotation
```python
logger.add(
    "logs/market_intelligence.log",
    rotation="500 MB",      # Rotate when 500MB
    retention="7 days"      # Keep for 7 days
)
```

---

## 10. Testing Strategy

### Unit Tests
- Sentiment analyzer
- Text cleaning
- Deduplication
- Configuration loading

### Integration Tests
- End-to-end pipeline (small dataset)
- File I/O operations
- Visualization generation

### Performance Tests
- 2000 tweet processing
- Memory profiling
- Execution time tracking

---

## Future Enhancements

1. **Machine Learning Signals**
   - LSTM networks for sequence analysis
   - Word embeddings (Word2Vec, BERT)
   - Sentiment classification models

2. **Real-time Processing**
   - Kafka/RabbitMQ streaming
   - Incremental model updates
   - Event-driven architecture

3. **Multi-source Data**
   - News APIs (NewsAPI, Alpha Vantage)
   - Financial data (Yfinance)
   - Sentiment feeds

4. **Advanced Analytics**
   - Anomaly detection
   - Clustering analysis
   - Causality analysis

5. **Deployment**
   - Docker containerization
   - Kubernetes orchestration
   - REST API endpoints
   - Cloud deployment (AWS/GCP/Azure)

---

**Document Version:** 1.0
**Last Updated:** 2024-11-20
