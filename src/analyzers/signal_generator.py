"""
Market Analysis Module
Converts textual data into quantitative signals for algorithmic trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import json

from loguru import logger

# ML/NLP imports
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    logger.warning("PyTorch/Transformers not installed. Install with: pip install torch transformers")


class FinBERTSentimentAnalyzer:
    """
    Fine-tuned FinBERT sentiment analyzer for financial market sentiment
    Uses pre-trained FinBERT model specifically tuned for finance domain
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize FinBERT sentiment analyzer
        
        Args:
            model_name: HuggingFace model ID (default: FinBERT)
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._initialize_model()
        
        # Index-specific keywords for post-processing
        self.index_keywords = {
            'nifty50': ['nifty50', 'nifty', '#nifty', 'nifty50-'],
            'sensex': ['sensex', '#sensex', 'bse', 'sensex-'],
            'banknifty': ['banknifty', 'bank nifty', '#banknifty', 'banknifty-']
        }
        
        logger.add("logs/analyzer.log", rotation="500 MB", retention="7 days")
    
    def _initialize_model(self):
        """Initialize FinBERT model and tokenizer"""
        try:
            if not BERT_AVAILABLE:
                logger.error("PyTorch/Transformers not available. Please install: pip install torch transformers")
                return False
            
            logger.info(f"Loading FinBERT model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info(f"FinBERT model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            logger.warning("Falling back to lexicon-based sentiment analysis")
            return False
    
    def calculate_sentiment_score(self, text: str) -> float:
        """
        Calculate sentiment using FinBERT
        
        Args:
            text: Input text/tweet
            
        Returns:
            Sentiment score [-1.0 (negative), 0.0 (neutral), 1.0 (positive)]
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        if self.pipeline is None:
            logger.warning("FinBERT model not available, using fallback")
            return self._fallback_sentiment(text)
        
        try:
            # Truncate to max length (512 tokens for BERT)
            text = text[:512]
            
            # Get prediction from FinBERT
            with torch.no_grad():
                result = self.pipeline(text)
            
            # FinBERT outputs: {'label': 'positive'/'negative'/'neutral', 'score': float}
            label = result[0]['label'].lower()
            score = result[0]['score']
            
            # Convert to [-1, 1] scale
            if label == 'positive':
                return min(score, 1.0)
            elif label == 'negative':
                return -min(score, 1.0)
            else:  # neutral
                return 0.0
                
        except Exception as e:
            logger.error(f"Error in FinBERT sentiment calculation: {e}")
            return self._fallback_sentiment(text)
    
    def _fallback_sentiment(self, text: str) -> float:
        """
        Fallback lexicon-based sentiment if FinBERT unavailable
        """
        positive_words = {
            'bullish': 0.8, 'bull': 0.7, 'buy': 0.6, 'long': 0.6,
            'breakout': 0.8, 'uptrend': 0.7, 'strong': 0.6,
            'rally': 0.7, 'surge': 0.7, 'green': 0.6,
        }
        
        negative_words = {
            'bearish': -0.8, 'bear': -0.7, 'sell': -0.6, 'short': -0.6,
            'breakdown': -0.8, 'downtrend': -0.7, 'weak': -0.6,
            'crash': -0.8, 'drop': -0.7, 'red': -0.6,
        }
        
        text_lower = text.lower()
        words = text_lower.split()
        
        score = sum(positive_words.get(w, 0) for w in words)
        score += sum(negative_words.get(w, 0) for w in words)
        
        return np.clip(score / 3.0, -1.0, 1.0)
    
    def get_index_sentiment(self, text: str, index: str) -> Optional[float]:
        """Get sentiment for specific index"""
        text_lower = text.lower()
        keywords = self.index_keywords.get(index.lower())
        
        if not keywords:
            return None
        
        if not any(kw in text_lower for kw in keywords):
            return None
        
        return self.calculate_sentiment_score(text)
    
    def get_sentiment_with_confidence(self, text: str) -> Dict[str, Any]:
        """Get sentiment with confidence score"""
        if not text or len(text.strip()) == 0:
            return {'sentiment': 0.0, 'confidence': 0.0, 'label': 'neutral'}
        
        if self.pipeline is None:
            return {'sentiment': self._fallback_sentiment(text), 'confidence': 0.5, 'label': 'fallback'}
        
        try:
            text = text[:512]
            with torch.no_grad():
                result = self.pipeline(text)
            
            label = result[0]['label'].lower()
            score = result[0]['score']
            
            if label == 'positive':
                sentiment = min(score, 1.0)
            elif label == 'negative':
                sentiment = -min(score, 1.0)
            else:
                sentiment = 0.0
            
            return {
                'sentiment': sentiment,
                'confidence': score,
                'label': label
            }
            
        except Exception as e:
            logger.error(f"Error: {e}")
            return {'sentiment': 0.0, 'confidence': 0.0, 'label': 'error'}


# Keep legacy analyzer as fallback
class LexiconSentimentAnalyzer:
    """Fallback lexicon-based analyzer if FinBERT fails"""
    
    def __init__(self):
        self.positive_words = {
            'bullish': 2.5, 'bull': 2.3, 'buy': 2.0, 'long': 1.8,
            'breakout': 2.3, 'uptrend': 2.2, 'strong': 1.8,
        }
        self.negative_words = {
            'bearish': -2.5, 'bear': -2.3, 'sell': -2.0, 'short': -1.8,
            'breakdown': -2.3, 'downtrend': -2.2, 'weak': -1.8,
        }
        self.index_keywords = {
            'nifty50': ['nifty50', 'nifty', '#nifty'],
            'sensex': ['sensex', '#sensex', 'bse'],
            'banknifty': ['banknifty', 'bank nifty', '#banknifty']
        }
    
    def calculate_sentiment_score(self, text: str) -> float:
        text_lower = text.lower()
        words = text_lower.split()
        
        score = sum(self.positive_words.get(w, 0) for w in words)
        score += sum(self.negative_words.get(w, 0) for w in words)
        
        return np.clip(score / 15.0, -1.0, 1.0)
    
    def get_index_sentiment(self, text: str, index: str) -> Optional[float]:
        text_lower = text.lower()
        keywords = self.index_keywords.get(index.lower())
        
        if not keywords:
            return None
        
        if not any(kw in text_lower for kw in keywords):
            return None
        
        return self.calculate_sentiment_score(text)


# Use FinBERT if available, fallback to lexicon
SentimentAnalyzer = FinBERTSentimentAnalyzer if BERT_AVAILABLE else LexiconSentimentAnalyzer





class TextFeatureExtractor:
    """Extracts numerical features from text data"""
    
    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self.tfidf_vectorizer = None
        self.feature_names = None
    
    def fit_tfidf(self, texts: List[str]) -> 'TextFeatureExtractor':
        """Fit TF-IDF vectorizer on texts"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.tfidf_vectorizer.fit(texts)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        logger.info(f"Fitted TF-IDF with {len(self.feature_names)} features")
        return self
    
    def transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """Transform texts using TF-IDF"""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf first.")
        return self.tfidf_vectorizer.transform(texts).toarray()
    
    def get_top_features(self, doc_index: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get top features for a document"""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted.")
        
        tfidf_matrix = self.tfidf_vectorizer.transform([])
        # This would need the actual matrix
        return []
    
    @staticmethod
    def extract_text_statistics(text: str) -> Dict[str, float]:
        """Extract statistical features from text"""
        words = text.split()
        sentences = text.split('.')
        
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        unique_words = len(set(word.lower() for word in words))
        
        return {
            'text_length': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': avg_word_length,
            'unique_word_ratio': unique_words / len(words) if words else 0,
            'punctuation_count': sum(1 for c in text if c in '!?.,:;'),
        }


class TradingSignalGenerator:
    """Generates trading signals from processed tweet data"""
    
    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self.sentiment_analyzer = SentimentAnalyzer()
        self.feature_extractor = TextFeatureExtractor()
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from DataFrame
        
        Args:
            df: DataFrame with tweet data
            
        Returns:
            DataFrame with trading signals
        """
        signals_df = df.copy()
        
        # Calculate sentiment scores
        signals_df['sentiment_score'] = signals_df['cleaned_content'].apply(
            self.sentiment_analyzer.calculate_sentiment_score
        )
        
        # Add index-specific sentiments
        signals_df['nifty50_sentiment'] = signals_df['cleaned_content'].apply(
            lambda x: self.sentiment_analyzer.get_index_sentiment(x, 'nifty50')
        )
        signals_df['sensex_sentiment'] = signals_df['cleaned_content'].apply(
            lambda x: self.sentiment_analyzer.get_index_sentiment(x, 'sensex')
        )
        signals_df['banknifty_sentiment'] = signals_df['cleaned_content'].apply(
            lambda x: self.sentiment_analyzer.get_index_sentiment(x, 'banknifty')
        )
        
        # Extract text statistics
        text_stats = signals_df['cleaned_content'].apply(
            TextFeatureExtractor.extract_text_statistics
        )
        stats_df = pd.json_normalize(text_stats)
        signals_df = pd.concat([signals_df, stats_df], axis=1)
        
        # Calculate composite signal
        signals_df['composite_signal'] = self._calculate_composite_signal(signals_df)
        
        # Calculate confidence intervals
        signals_df['signal_confidence'] = self._calculate_confidence(signals_df)
        
        # Generate signal type
        signals_df['signal_type'] = signals_df.apply(self._determine_signal_type, axis=1)
        
        # Calculate weighted signal strength
        signals_df['signal_strength'] = signals_df.apply(self._calculate_signal_strength, axis=1)
        
        # Generate index-specific signals
        signals_df['nifty50_signal'] = signals_df.apply(lambda x: self._generate_index_signal(x, 'nifty50'), axis=1)
        signals_df['sensex_signal'] = signals_df.apply(lambda x: self._generate_index_signal(x, 'sensex'), axis=1)
        signals_df['banknifty_signal'] = signals_df.apply(lambda x: self._generate_index_signal(x, 'banknifty'), axis=1)
        
        logger.info(f"Generated signals for {len(signals_df)} tweets")
        return signals_df
    
    def _generate_index_signal(self, row, index: str) -> str:
        """Generate specific buy/sell signal for an index"""
        sentiment = row.get(f'{index}_sentiment')
        
        # Use overall sentiment if index-specific not available
        if pd.isna(sentiment) or sentiment is None:
            sentiment = row.get('sentiment_score', 0)
        
        # Lower thresholds for better small-sample detection
        if sentiment > 0.1:
            return 'BUY'
        elif sentiment < -0.1:
            return 'SELL'
        elif sentiment > 0.02:
            return 'HOLD'
        else:
            return 'HOLD'
    
    def _calculate_composite_signal(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate composite trading signal
        Combines multiple features into single signal
        """
        signal = np.zeros(len(df))
        
        # Sentiment component (increased weight)
        sentiment_weight = 0.4
        signal += df['sentiment_score'] * sentiment_weight
        
        # Engagement component
        engagement_weight = 0.2
        normalized_engagement = (df['total_engagement'] - df['total_engagement'].min()) / \
                               (df['total_engagement'].max() - df['total_engagement'].min() + 1e-8)
        signal += normalized_engagement * engagement_weight
        
        # Market signal component
        market_signal_weight = 0.2
        buy_sell_diff = (df['mentions_buy_terms'] - df['mentions_sell_terms']) / \
                       (df['mentions_buy_terms'] + df['mentions_sell_terms'] + 1e-8)
        signal += buy_sell_diff * market_signal_weight
        
        # Bullish/Bearish component
        bullish_bearish_weight = 0.2
        bullish_bearish_diff = (df['mentions_bullish_terms'] - df['mentions_bearish_terms']) / \
                              (df['mentions_bullish_terms'] + df['mentions_bearish_terms'] + 1e-8)
        signal += bullish_bearish_diff * bullish_bearish_weight
        
        # Normalize to [-1, 1]
        signal = np.tanh(signal)
        
        return signal
    
    def _calculate_confidence(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate confidence intervals for signals"""
        confidence = np.zeros(len(df))
        
        # Base confidence from engagement (increased from 0.4 to 0.5)
        max_engagement = df['total_engagement'].max() if df['total_engagement'].max() > 0 else 1
        if max_engagement > 0:
            engagement_conf = np.clip(df['total_engagement'] / max_engagement, 0, 1)
        else:
            engagement_conf = np.full(len(df), 0.5)  # Default for small samples
        confidence += engagement_conf * 0.45
        
        # Confidence from text quality
        text_quality_conf = df['unique_word_ratio'] * 0.35
        confidence += text_quality_conf
        
        # Confidence from sentiment strength (new: direct sentiment magnitude)
        if 'sentiment_score' in df.columns:
            sentiment_conf = np.abs(df['sentiment_score']) * 0.2
            confidence += sentiment_conf
        
        return np.clip(confidence, 0.3, 1.0)  # Floor at 0.3 for small samples
    
    def _determine_signal_type(self, row) -> str:
        """Determine signal type (BUY, SELL, HOLD, NEUTRAL)"""
        signal_strength = abs(row['composite_signal'])
        confidence = row['signal_confidence']
        
        # Lower thresholds for small samples
        min_threshold = 0.2  # Reduced from 0.5
        strength_threshold = 0.25  # Reduced from 0.35
        
        if confidence < min_threshold:
            return 'NEUTRAL'
        
        if row['composite_signal'] > strength_threshold:
            return 'BUY'
        elif row['composite_signal'] < -strength_threshold:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _calculate_signal_strength(self, row) -> float:
        """Calculate signal strength [0, 1]"""
        return abs(row['composite_signal']) * row['signal_confidence']
    
    def aggregate_signals(self, signals_df: pd.DataFrame, period: str = '1H') -> pd.DataFrame:
        """
        Aggregate signals over time periods
        
        Args:
            signals_df: DataFrame with signals
            period: Time period (e.g., '1H', '1D')
            
        Returns:
            Aggregated signals by time period
        """
        if 'timestamp' not in signals_df.columns:
            logger.warning("No timestamp column found")
            return signals_df
        
        # Convert timestamp to datetime
        signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'], errors='coerce')
        
        # Group by time period
        aggregated = signals_df.groupby(pd.Grouper(key='timestamp', freq=period)).agg({
            'composite_signal': ['mean', 'std', 'count'],
            'signal_confidence': 'mean',
            'total_engagement': 'sum',
            'sentiment_score': 'mean',
            'mentions_buy_terms': 'sum',
            'mentions_sell_terms': 'sum',
        }).reset_index()
        
        aggregated.columns = ['_'.join(col).strip('_') for col in aggregated.columns]
        
        logger.info(f"Aggregated into {len(aggregated)} time periods")
        return aggregated
    
    def filter_strong_signals(self, signals_df: pd.DataFrame, 
                            min_strength: float = 0.7) -> pd.DataFrame:
        """Filter signals by strength threshold"""
        filtered = signals_df[signals_df['signal_strength'] >= min_strength].copy()
        logger.info(f"Filtered to {len(filtered)} strong signals out of {len(signals_df)}")
        return filtered


class InsightGenerator:
    """Generates actionable insights from signals"""
    
    @staticmethod
    def get_top_signals(signals_df: pd.DataFrame, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get top trading signals by strength"""
        top_signals = signals_df.nlargest(top_n, 'signal_strength')
        
        insights = []
        for idx, row in top_signals.iterrows():
            insight = {
                'rank': len(insights) + 1,
                'signal_type': row['signal_type'],
                'strength': float(row['signal_strength']),
                'confidence': float(row['signal_confidence']),
                'content': row['cleaned_content'][:100],
                'sentiment': float(row['sentiment_score']),
                'engagement': int(row['total_engagement']),
                'timestamp': str(row['timestamp']),
            }
            insights.append(insight)
        
        return insights
    
    @staticmethod
    def get_market_summary(signals_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate overall market summary with index-specific analysis"""
        buy_signals = len(signals_df[signals_df['signal_type'] == 'BUY'])
        sell_signals = len(signals_df[signals_df['signal_type'] == 'SELL'])
        hold_signals = len(signals_df[signals_df['signal_type'] == 'HOLD'])
        neutral_signals = len(signals_df[signals_df['signal_type'] == 'NEUTRAL'])
        
        # Index-specific analysis
        nifty50_buy = len(signals_df[signals_df['nifty50_signal'] == 'BUY'])
        nifty50_sell = len(signals_df[signals_df['nifty50_signal'] == 'SELL'])
        nifty50_hold = len(signals_df[signals_df['nifty50_signal'] == 'HOLD'])
        
        sensex_buy = len(signals_df[signals_df['sensex_signal'] == 'BUY'])
        sensex_sell = len(signals_df[signals_df['sensex_signal'] == 'SELL'])
        sensex_hold = len(signals_df[signals_df['sensex_signal'] == 'HOLD'])
        
        banknifty_buy = len(signals_df[signals_df['banknifty_signal'] == 'BUY'])
        banknifty_sell = len(signals_df[signals_df['banknifty_signal'] == 'SELL'])
        banknifty_hold = len(signals_df[signals_df['banknifty_signal'] == 'HOLD'])
        
        # Average sentiments - use overall sentiment if index-specific is unavailable
        overall_sentiment = signals_df['sentiment_score'].mean()
        nifty50_sentiment = signals_df['nifty50_sentiment'].dropna().mean() if len(signals_df['nifty50_sentiment'].dropna()) > 0 else overall_sentiment
        sensex_sentiment = signals_df['sensex_sentiment'].dropna().mean() if len(signals_df['sensex_sentiment'].dropna()) > 0 else overall_sentiment
        banknifty_sentiment = signals_df['banknifty_sentiment'].dropna().mean() if len(signals_df['banknifty_sentiment'].dropna()) > 0 else overall_sentiment
        
        # Determine market direction
        total_buy = nifty50_buy + sensex_buy + banknifty_buy
        total_sell = nifty50_sell + sensex_sell + banknifty_sell
        
        if total_buy > total_sell * 1.5:
            market_direction = 'STRONGLY BULLISH'
        elif total_buy > total_sell:
            market_direction = 'BULLISH'
        elif total_sell > total_buy * 1.5:
            market_direction = 'STRONGLY BEARISH'
        elif total_sell > total_buy:
            market_direction = 'BEARISH'
        else:
            market_direction = 'NEUTRAL'
        
        summary = {
            'overall_market': {
                'total_signals': len(signals_df),
                'market_direction': market_direction,
                'buy_count': int(buy_signals),
                'sell_count': int(sell_signals),
                'hold_count': int(hold_signals),
                'neutral_count': int(neutral_signals),
                'buy_percentage': float(buy_signals / len(signals_df) * 100) if len(signals_df) > 0 else 0,
                'sell_percentage': float(sell_signals / len(signals_df) * 100) if len(signals_df) > 0 else 0,
                'avg_signal_strength': float(signals_df['signal_strength'].mean()),
                'avg_confidence': float(signals_df['signal_confidence'].mean()),
                'avg_sentiment': float(signals_df['sentiment_score'].mean()),
            },
            'nifty50': {
                'signal': 'BUY' if nifty50_buy > nifty50_sell else 'SELL' if nifty50_sell > nifty50_buy else 'NEUTRAL',
                'sentiment': float(nifty50_sentiment),
                'buy_signals': int(nifty50_buy),
                'sell_signals': int(nifty50_sell),
                'hold_signals': int(nifty50_hold),
                'bullish_count': int(nifty50_buy),
                'bearish_count': int(nifty50_sell),
                'conviction': 'HIGH' if (abs(nifty50_sentiment) > 0.15 or nifty50_buy > len(signals_df) * 0.15) else 'MEDIUM' if (abs(nifty50_sentiment) > 0.05 or nifty50_buy > len(signals_df) * 0.05) else 'LOW',
            },
            'sensex': {
                'signal': 'BUY' if sensex_buy > sensex_sell else 'SELL' if sensex_sell > sensex_buy else 'NEUTRAL',
                'sentiment': float(sensex_sentiment),
                'buy_signals': int(sensex_buy),
                'sell_signals': int(sensex_sell),
                'hold_signals': int(sensex_hold),
                'bullish_count': int(sensex_buy),
                'bearish_count': int(sensex_sell),
                'conviction': 'HIGH' if (abs(sensex_sentiment) > 0.15 or sensex_buy > len(signals_df) * 0.15) else 'MEDIUM' if (abs(sensex_sentiment) > 0.05 or sensex_buy > len(signals_df) * 0.05) else 'LOW',
            },
            'banknifty': {
                'signal': 'BUY' if banknifty_buy > banknifty_sell else 'SELL' if banknifty_sell > banknifty_buy else 'NEUTRAL',
                'sentiment': float(banknifty_sentiment),
                'buy_signals': int(banknifty_buy),
                'sell_signals': int(banknifty_sell),
                'hold_signals': int(banknifty_hold),
                'bullish_count': int(banknifty_buy),
                'bearish_count': int(banknifty_sell),
                'conviction': 'HIGH' if (abs(banknifty_sentiment) > 0.15 or banknifty_buy > len(signals_df) * 0.15) else 'MEDIUM' if (abs(banknifty_sentiment) > 0.05 or banknifty_buy > len(signals_df) * 0.05) else 'LOW',
            },
        }
        
        return summary
    
    @staticmethod
    def get_index_signals(signals_df: pd.DataFrame, index: str = 'nifty50') -> Dict[str, Any]:
        """Get detailed signals for specific index"""
        signal_col = f'{index}_signal'
        sentiment_col = f'{index}_sentiment'
        
        if signal_col not in signals_df.columns:
            return {'error': f'Index {index} not found'}
        
        buy_signals = signals_df[signals_df[signal_col] == 'BUY'].nlargest(5, 'signal_strength')
        sell_signals = signals_df[signals_df[signal_col] == 'SELL'].nlargest(5, 'signal_strength')
        
        buy_insights = [
            {
                'type': 'BUY',
                'content': row['cleaned_content'][:80],
                'sentiment': float(row[sentiment_col]),
                'strength': float(row['signal_strength']),
                'engagement': int(row['total_engagement']),
            }
            for _, row in buy_signals.iterrows()
        ]
        
        sell_insights = [
            {
                'type': 'SELL',
                'content': row['cleaned_content'][:80],
                'sentiment': float(row[sentiment_col]),
                'strength': float(row['signal_strength']),
                'engagement': int(row['total_engagement']),
            }
            for _, row in sell_signals.iterrows()
        ]
        
        return {
            'index': index.upper(),
            'top_buy_signals': buy_insights,
            'top_sell_signals': sell_insights,
        }
