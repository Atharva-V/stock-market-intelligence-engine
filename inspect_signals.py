#!/usr/bin/env python3
import pandas as pd
import numpy as np
from src.analyzers.signal_generator import TradingSignalGenerator

# Load data
df = pd.read_csv('data/processed/market_data.csv')
print(f"Loaded {len(df)} tweets\n")

# Generate signals
engine = TradingSignalGenerator()
signals_df = engine.generate_signals(df)

# Show key columns for top BUY signals
buy_signals = signals_df[signals_df['signal_type'] == 'BUY'].sort_values('signal_strength', ascending=False)
print(f"Found {len(buy_signals)} BUY signals\n")

print("Top BUY signals details:")
print("-" * 100)
for idx, (i, row) in enumerate(buy_signals.head(5).iterrows(), 1):
    print(f"\n{idx}. BUY Signal (Strength: {row['signal_strength']:.2%}, Confidence: {row['signal_confidence']:.2%})")
    print(f"   Sentiment: {row['sentiment_score']:.3f}")
    print(f"   Nifty50 Sentiment: {row['nifty50_sentiment']}")
    print(f"   Nifty50 Signal: {row['nifty50_signal']}")
    print(f"   Content: {row['cleaned_content'][:80]}...")

print("\n" + "=" * 100)
print("\nAll signals sentiment range:")
print(f"  Min: {signals_df['sentiment_score'].min():.3f}")
print(f"  Max: {signals_df['sentiment_score'].max():.3f}")
print(f"  Mean: {signals_df['sentiment_score'].mean():.3f}")
print(f"  Median: {signals_df['sentiment_score'].median():.3f}")

print("\nSignal type distribution:")
print(signals_df['signal_type'].value_counts())
