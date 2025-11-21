"""
Visualization Module
Memory-efficient plotting solutions for large datasets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from loguru import logger


class EfficientVisualizer:
    """
    Memory-efficient visualization for large datasets
    Uses sampling techniques and streaming plots
    """
    
    def __init__(self, output_path: str = "output/visualizations", figsize: Tuple[int, int] = (12, 6)):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = figsize
        
        logger.add("logs/visualizer.log", rotation="500 MB", retention="7 days")
    
    def plot_signal_distribution(self, signals_df: pd.DataFrame, 
                                sample_size: Optional[int] = None) -> str:
        """
        Plot distribution of trading signals
        
        Args:
            signals_df: DataFrame with signals
            sample_size: Sample size for large datasets
            
        Returns:
            Path to saved figure
        """
        # Sample if needed
        if sample_size and len(signals_df) > sample_size:
            df_sample = signals_df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampling {sample_size} records from {len(signals_df)}")
        else:
            df_sample = signals_df
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Signal type distribution
        signal_counts = df_sample['signal_type'].value_counts()
        colors = {'BUY': '#2ecc71', 'SELL': '#e74c3c', 'HOLD': '#f39c12', 'NEUTRAL': '#95a5a6'}
        signal_colors = [colors.get(sig, '#95a5a6') for sig in signal_counts.index]
        
        axes[0, 0].bar(signal_counts.index, signal_counts.values, color=signal_colors)
        axes[0, 0].set_title('Trading Signal Distribution')
        axes[0, 0].set_ylabel('Count')
        
        # Composite signal distribution
        axes[0, 1].hist(df_sample['composite_signal'], bins=50, color='#3498db', edgecolor='black')
        axes[0, 1].set_title('Composite Signal Distribution')
        axes[0, 1].set_xlabel('Signal Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        
        # Signal strength vs confidence
        scatter = axes[1, 0].scatter(df_sample['signal_confidence'], 
                                    df_sample['signal_strength'],
                                    c=df_sample['composite_signal'],
                                    cmap='RdYlGn', alpha=0.6, s=30)
        axes[1, 0].set_title('Signal Strength vs Confidence')
        axes[1, 0].set_xlabel('Confidence')
        axes[1, 0].set_ylabel('Strength')
        plt.colorbar(scatter, ax=axes[1, 0], label='Composite Signal')
        
        # Sentiment distribution
        axes[1, 1].hist(df_sample['sentiment_score'], bins=50, color='#9b59b6', edgecolor='black')
        axes[1, 1].set_title('Sentiment Score Distribution')
        axes[1, 1].set_xlabel('Sentiment Score')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        output_file = self.output_path / "signal_distribution.png"
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved signal distribution plot to {output_file}")
        return str(output_file)
    
    def plot_engagement_analysis(self, signals_df: pd.DataFrame, 
                                sample_size: Optional[int] = None) -> str:
        """Plot engagement metrics analysis"""
        if sample_size and len(signals_df) > sample_size:
            df_sample = signals_df.sample(n=sample_size, random_state=42)
        else:
            df_sample = signals_df
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Engagement by signal type
        engagement_by_type = df_sample.groupby('signal_type')['total_engagement'].mean()
        axes[0, 0].bar(engagement_by_type.index, engagement_by_type.values, 
                      color=['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6'][:len(engagement_by_type)])
        axes[0, 0].set_title('Average Engagement by Signal Type')
        axes[0, 0].set_ylabel('Engagement Score')
        
        # Engagement distribution
        axes[0, 1].hist(df_sample['total_engagement'], bins=50, color='#3498db', edgecolor='black')
        axes[0, 1].set_title('Total Engagement Distribution')
        axes[0, 1].set_xlabel('Engagement')
        axes[0, 1].set_ylabel('Frequency')
        
        # Likes vs retweets
        axes[1, 0].scatter(df_sample['likes'], df_sample['retweets'], 
                          alpha=0.5, s=30, color='#e74c3c')
        axes[1, 0].set_title('Likes vs Retweets')
        axes[1, 0].set_xlabel('Likes (normalized)')
        axes[1, 0].set_ylabel('Retweets (normalized)')
        
        # Engagement score distribution
        axes[1, 1].hist(df_sample['engagement_score'], bins=50, color='#f39c12', edgecolor='black')
        axes[1, 1].set_title('Engagement Score Distribution')
        axes[1, 1].set_xlabel('Score')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        output_file = self.output_path / "engagement_analysis.png"
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved engagement analysis to {output_file}")
        return str(output_file)
    
    def plot_temporal_analysis(self, signals_df: pd.DataFrame) -> str:
        """Plot temporal patterns in signals"""
        if 'timestamp' not in signals_df.columns:
            logger.warning("No timestamp column found")
            return ""
        
        df_time = signals_df.copy()
        df_time['timestamp'] = pd.to_datetime(df_time['timestamp'], errors='coerce')
        df_time = df_time.dropna(subset=['timestamp'])
        
        if len(df_time) == 0:
            logger.warning("No valid timestamps")
            return ""
        
        # Resample by hour
        df_hourly = df_time.set_index('timestamp').resample('H').agg({
            'composite_signal': 'mean',
            'signal_confidence': 'mean',
            'total_engagement': 'sum',
        }).reset_index()
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Composite signal over time
        axes[0].plot(df_hourly['timestamp'], df_hourly['composite_signal'], 
                    marker='o', color='#3498db', linewidth=2)
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0].fill_between(df_hourly['timestamp'], 
                            df_hourly['composite_signal'], 
                            alpha=0.3, color='#3498db')
        axes[0].set_title('Composite Signal Over Time')
        axes[0].set_ylabel('Signal Value')
        axes[0].grid(True, alpha=0.3)
        
        # Confidence over time
        axes[1].plot(df_hourly['timestamp'], df_hourly['signal_confidence'], 
                    marker='s', color='#2ecc71', linewidth=2)
        axes[1].set_title('Signal Confidence Over Time')
        axes[1].set_ylabel('Confidence')
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3)
        
        # Engagement over time
        axes[2].bar(df_hourly['timestamp'], df_hourly['total_engagement'], 
                   color='#e74c3c', alpha=0.7)
        axes[2].set_title('Total Engagement Over Time')
        axes[2].set_xlabel('Timestamp')
        axes[2].set_ylabel('Engagement')
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_file = self.output_path / "temporal_analysis.png"
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved temporal analysis to {output_file}")
        return str(output_file)
    
    def plot_sentiment_analysis(self, signals_df: pd.DataFrame, 
                               sample_size: Optional[int] = None) -> str:
        """Plot sentiment-related metrics"""
        if sample_size and len(signals_df) > sample_size:
            df_sample = signals_df.sample(n=sample_size, random_state=42)
        else:
            df_sample = signals_df
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Sentiment by signal type
        sentiment_by_type = df_sample.groupby('signal_type')['sentiment_score'].mean()
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']
        axes[0, 0].bar(sentiment_by_type.index, sentiment_by_type.values, 
                      color=colors[:len(sentiment_by_type)])
        axes[0, 0].set_title('Average Sentiment by Signal Type')
        axes[0, 0].set_ylabel('Sentiment Score')
        axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Sentiment distribution
        axes[0, 1].hist(df_sample['sentiment_score'], bins=50, color='#9b59b6', edgecolor='black')
        axes[0, 1].set_title('Sentiment Distribution')
        axes[0, 1].set_xlabel('Sentiment Score')
        axes[0, 1].set_ylabel('Frequency')
        
        # Market terms comparison
        market_terms_data = {
            'Buy Terms': df_sample['mentions_buy_terms'].sum(),
            'Sell Terms': df_sample['mentions_sell_terms'].sum(),
            'Bullish Terms': df_sample['mentions_bullish_terms'].sum(),
            'Bearish Terms': df_sample['mentions_bearish_terms'].sum(),
        }
        
        colors_terms = ['#2ecc71', '#e74c3c', '#3498db', '#e67e22']
        axes[1, 0].bar(market_terms_data.keys(), market_terms_data.values(), 
                      color=colors_terms)
        axes[1, 0].set_title('Market-Related Terms Count')
        axes[1, 0].set_ylabel('Count')
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Sentiment vs engagement
        scatter = axes[1, 1].scatter(df_sample['sentiment_score'], 
                                    df_sample['total_engagement'],
                                    c=df_sample['composite_signal'],
                                    cmap='RdYlGn', alpha=0.6, s=30)
        axes[1, 1].set_title('Sentiment vs Engagement')
        axes[1, 1].set_xlabel('Sentiment Score')
        axes[1, 1].set_ylabel('Total Engagement')
        plt.colorbar(scatter, ax=axes[1, 1], label='Composite Signal')
        
        plt.tight_layout()
        
        output_file = self.output_path / "sentiment_analysis.png"
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved sentiment analysis to {output_file}")
        return str(output_file)
    
    def plot_hashtag_frequency(self, signals_df: pd.DataFrame, top_n: int = 20) -> str:
        """Plot top hashtags frequency"""
        # Extract and count hashtags
        hashtags = signals_df['hashtags'].explode().dropna()
        top_hashtags = hashtags.value_counts().head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Handle empty hashtags case
        if len(top_hashtags) == 0:
            ax.text(0.5, 0.5, 'No hashtags found in collected tweets', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title(f'Top {top_n} Hashtags')
            ax.axis('off')
        else:
            top_hashtags.plot(kind='barh', ax=ax, color='#3498db')
            ax.set_title(f'Top {top_n} Hashtags')
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Hashtag')
        
        plt.tight_layout()
        
        output_file = self.output_path / "hashtag_frequency.png"
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved hashtag frequency to {output_file}")
        return str(output_file)
    
    def create_summary_report(self, signals_df: pd.DataFrame, 
                             market_summary: Dict, 
                             top_signals: List[Dict]) -> str:
        """Create visual summary report"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Market Intelligence Summary Report', fontsize=16, fontweight='bold')
        
        # Signal distribution pie chart
        ax1 = fig.add_subplot(gs[0, 0])
        signal_counts = signals_df['signal_type'].value_counts()
        colors = {'BUY': '#2ecc71', 'SELL': '#e74c3c', 'HOLD': '#f39c12', 'NEUTRAL': '#95a5a6'}
        signal_colors = [colors.get(sig, '#95a5a6') for sig in signal_counts.index]
        ax1.pie(signal_counts.values, labels=signal_counts.index, autopct='%1.1f%%',
               colors=signal_colors, startangle=90)
        ax1.set_title('Signal Distribution')
        
        # Market metrics
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        
        # Handle both old and new market_summary structures
        if 'overall_market' in market_summary:
            overall = market_summary['overall_market']
        else:
            overall = market_summary
        
        metrics_text = f"""
        Market Summary
        ─────────────────
        Total Signals: {overall.get('total_signals', len(signals_df))}
        Buy Signals: {overall.get('buy_count', 0)}
        Sell Signals: {overall.get('sell_count', 0)}
        Avg Confidence: {overall.get('avg_confidence', 0):.2%}
        Market Direction: {overall.get('market_direction', 'UNKNOWN')}
        """
        
        ax2.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Top signals
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')
        
        top_signals_text = "Top Trading Signals\n" + "─" * 80 + "\n"
        for i, signal in enumerate(top_signals[:5], 1):
            top_signals_text += f"{i}. [{signal['signal_type']}] Strength: {signal['strength']:.2%} | "
            top_signals_text += f"Confidence: {signal['confidence']:.2%} | Sentiment: {signal['sentiment']:.2f}\n"
        
        ax3.text(0.05, 0.5, top_signals_text, fontsize=9, family='monospace',
                verticalalignment='center')
        
        # Composite signal distribution
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.hist(signals_df['composite_signal'], bins=40, color='#3498db', edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax4.set_title('Composite Signal Distribution')
        ax4.set_xlabel('Signal Value')
        ax4.set_ylabel('Frequency')
        
        # Confidence distribution
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.hist(signals_df['signal_confidence'], bins=40, color='#2ecc71', edgecolor='black')
        ax5.set_title('Confidence Distribution')
        ax5.set_xlabel('Confidence')
        ax5.set_ylabel('Frequency')
        
        output_file = self.output_path / "summary_report.png"
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved summary report to {output_file}")
        return str(output_file)
