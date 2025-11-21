"""
Debug script for signal generation and visualization
Uses existing processed data without scraping
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from src.analyzers.signal_generator import TradingSignalGenerator, InsightGenerator
from src.analyzers.visualization import EfficientVisualizer
from src.utils.config import Config
from loguru import logger

logger.add("logs/debug.log", rotation="500 MB", retention="7 days")


def main():
    print("\n" + "="*70)
    print("DEBUG: Signal Generation & Visualization")
    print("="*70)
    
    config = Config()
    
    # Step 1: Load existing processed data
    print("\n📂 STEP 1: Loading Processed Data")
    print("-" * 70)
    
    csv_file = Path("data/processed/market_data.csv")
    parquet_file = Path("data/processed/market_data.parquet")
    
    if csv_file.exists():
        print(f"Loading from CSV: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"✓ Loaded {len(df)} records from CSV")
    elif parquet_file.exists():
        print(f"Loading from Parquet: {parquet_file}")
        df = pd.read_parquet(parquet_file)
        print(f"✓ Loaded {len(df)} records from Parquet")
    else:
        print("❌ No processed data found!")
        print(f"   Expected: {csv_file} or {parquet_file}")
        return 1
    
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Step 2: Generate signals
    print("\n" + "="*70)
    print("📈 STEP 2: Signal Generation")
    print("="*70)
    
    try:
        logger.info("Starting signal generation")
        signal_gen = TradingSignalGenerator(min_confidence=0.5)
        signals_df = signal_gen.generate_signals(df)
        
        print(f"✓ Generated signals for {len(signals_df)} records")
        
        # Show signal breakdown
        signal_counts = signals_df['signal_type'].value_counts()
        print(f"\nSignal Distribution:")
        for signal_type, count in signal_counts.items():
            pct = (count / len(signals_df)) * 100
            print(f"  - {signal_type}: {count} ({pct:.1f}%)")
        
        # Show sentiment distribution
        print(f"\nSentiment Statistics:")
        print(f"  - Mean: {signals_df['sentiment_score'].mean():.3f}")
        print(f"  - Std: {signals_df['sentiment_score'].std():.3f}")
        print(f"  - Min: {signals_df['sentiment_score'].min():.3f}")
        print(f"  - Max: {signals_df['sentiment_score'].max():.3f}")
        
        logger.info(f"Signal generation completed successfully")
        
    except Exception as e:
        print(f"❌ Error during signal generation: {e}")
        logger.error(f"Signal generation failed: {e}", exc_info=True)
        return 1
    
    # Step 3: Get insights
    print("\n" + "="*70)
    print("🔍 STEP 3: Market Analysis")
    print("="*70)
    
    try:
        insight_gen = InsightGenerator()
        market_summary = insight_gen.get_market_summary(signals_df)
        top_signals = insight_gen.get_top_signals(signals_df, top_n=10)
        
        # Access nested structure
        overall = market_summary['overall_market']
        
        print(f"\nOverall Market:")
        print(f"  - Market Direction: {overall['market_direction']}")
        print(f"  - Buy Signals: {overall['buy_count']} ({overall['buy_percentage']:.1f}%)")
        print(f"  - Sell Signals: {overall['sell_count']} ({overall['sell_percentage']:.1f}%)")
        print(f"  - Hold Signals: {overall['hold_count']}")
        print(f"  - Avg Sentiment: {overall['avg_sentiment']:.3f}")
        print(f"  - Avg Confidence: {overall['avg_confidence']:.2%}")
        print(f"  - Avg Signal Strength: {overall['avg_signal_strength']:.2%}")
        
        print(f"\nIndex-Specific Signals:")
        for index in ['nifty50', 'sensex', 'banknifty']:
            idx_data = market_summary[index]
            print(f"  {index.upper()}:")
            print(f"    - Signal: {idx_data['signal']}")
            print(f"    - Sentiment: {idx_data['sentiment']:.3f}")
            print(f"    - Conviction: {idx_data['conviction']}")
            print(f"    - Buy: {idx_data['buy_signals']}, Sell: {idx_data['sell_signals']}")
        
        print(f"\nTop 3 Strongest Signals:")
        for i, signal in enumerate(top_signals[:3], 1):
            print(f"  {i}. {signal['signal_type']} (Strength: {signal['strength']:.2%}, Confidence: {signal['confidence']:.2%})")
            print(f"     Content: {signal['content'][:60]}...")
        
        logger.info(f"Market analysis completed successfully")
        
    except Exception as e:
        print(f"❌ Error during market analysis: {e}")
        logger.error(f"Market analysis failed: {e}", exc_info=True)
        return 1
    
    # Step 4: Generate visualizations
    print("\n" + "="*70)
    print("📊 STEP 4: Visualization")
    print("="*70)
    
    try:
        visualizer = EfficientVisualizer(output_path="output/visualizations")
        viz_files = []
        
        print("Generating plots...")
        
        # 1. Signal distribution
        try:
            viz1 = visualizer.plot_signal_distribution(signals_df, 
                                                       sample_size=config.VISUALIZATION_SAMPLE_SIZE)
            viz_files.append(viz1)
            print(f"  ✓ Signal distribution: {viz1}")
        except Exception as e:
            print(f"  ⚠ Signal distribution failed: {e}")
            logger.warning(f"Signal distribution plot failed: {e}")
        
        # 2. Engagement analysis
        try:
            viz2 = visualizer.plot_engagement_analysis(signals_df,
                                                       sample_size=config.VISUALIZATION_SAMPLE_SIZE)
            viz_files.append(viz2)
            print(f"  ✓ Engagement analysis: {viz2}")
        except Exception as e:
            print(f"  ⚠ Engagement analysis failed: {e}")
            logger.warning(f"Engagement analysis plot failed: {e}")
        
        # 3. Sentiment analysis
        try:
            viz3 = visualizer.plot_sentiment_analysis(signals_df,
                                                      sample_size=config.VISUALIZATION_SAMPLE_SIZE)
            viz_files.append(viz3)
            print(f"  ✓ Sentiment analysis: {viz3}")
        except Exception as e:
            print(f"  ⚠ Sentiment analysis failed: {e}")
            logger.warning(f"Sentiment analysis plot failed: {e}")
        
        # 4. Temporal analysis
        try:
            viz4 = visualizer.plot_temporal_analysis(signals_df)
            if viz4:
                viz_files.append(viz4)
                print(f"  ✓ Temporal analysis: {viz4}")
            else:
                print(f"  ⚠ Temporal analysis: No timestamp data")
        except Exception as e:
            print(f"  ⚠ Temporal analysis failed: {e}")
            logger.warning(f"Temporal analysis plot failed: {e}")
        
        # 5. Hashtag frequency
        try:
            viz5 = visualizer.plot_hashtag_frequency(signals_df, top_n=20)
            viz_files.append(viz5)
            print(f"  ✓ Hashtag frequency: {viz5}")
        except Exception as e:
            print(f"  ⚠ Hashtag frequency failed: {e}")
            logger.warning(f"Hashtag frequency plot failed: {e}")
        
        # 6. Summary report
        try:
            viz6 = visualizer.create_summary_report(signals_df, market_summary, top_signals)
            viz_files.append(viz6)
            print(f"  ✓ Summary report: {viz6}")
        except Exception as e:
            print(f"  ⚠ Summary report failed: {e}")
            logger.warning(f"Summary report failed: {e}")
        
        print(f"\n✓ Created {len(viz_files)} visualizations")
        logger.info(f"Visualization completed: {len(viz_files)} files generated")
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        logger.error(f"Visualization failed: {e}", exc_info=True)
        return 1
    
    # Step 5: Save debug report
    print("\n" + "="*70)
    print("💾 STEP 5: Saving Debug Report")
    print("="*70)
    
    try:
        debug_report = {
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_records': len(signals_df),
                'columns': list(signals_df.columns),
            },
            'market_analysis': market_summary,
            'top_signals': top_signals,
            'visualizations': viz_files,
        }
        
        report_file = Path("output/debug_report.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(debug_report, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Debug report saved: {report_file}")
        logger.info(f"Debug report saved to {report_file}")
        
    except Exception as e:
        print(f"❌ Error saving debug report: {e}")
        logger.error(f"Debug report save failed: {e}", exc_info=True)
        return 1
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Processed {len(signals_df)} tweets")
    print(f"✓ Generated market signals")
    print(f"✓ Created {len(viz_files)} visualizations")
    print(f"✓ Saved debug report to {report_file}")
    print(f"\nOutput files in: output/visualizations/")
    print(f"Report: output/debug_report.json")
    print("\n" + "="*70)
    
    logger.info("Debug session completed successfully")
    print("\n✅ Debug session completed!\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
