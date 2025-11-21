"""
Main Orchestration Script
Coordinates data collection, processing, analysis, and reporting
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import numpy as np

from loguru import logger

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.scrapers.twitter_scraper import TwitterScraper, SharedBrowserScraper, MultiSessionTwitterScraper
from src.processors.data_processor import DataProcessor
from src.analyzers.signal_generator import TradingSignalGenerator, InsightGenerator
from src.analyzers.visualization import EfficientVisualizer
from src.utils.config import Config, setup_logging


def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("MARKET INTELLIGENCE SYSTEM - Real-time Data Collection & Analysis")
    print("="*70 + "\n")
    
    # Initialize configuration
    config = Config()
    setup_logging(config.LOG_FILE, config.LOG_LEVEL)
    
    logger.info("Starting Market Intelligence System")
    
    try:
        # Step 1: Data Collection
        print("\n📊 STEP 1: Data Collection from Twitter/X")
        print("-" * 70)
        logger.info("Starting data collection phase")
        
        # Use single browser, sequential scraping by hashtag
        print("Scraping tweets sequentially by hashtag (single browser session)...")
        scraper = TwitterScraper(
            headless=False,  # Show browser so you can see login
            rate_limit=0.2   # Very fast rate limit for speed
        )
        
        logger.info(f"Scraping {len(config.SEARCH_KEYWORDS)} queries sequentially")
        print(f"Target Keywords: {', '.join(config.SEARCH_KEYWORDS)}")
        print(f"Total tweets to collect: {config.TARGET_TWEET_COUNT}")
        
        # Scrape all queries sequentially using single browser session
        raw_tweets = []
        tweets_per_query = 300  # 300 per hashtag (7 hashtags = 2100+ total)
        
        for i, keyword in enumerate(config.SEARCH_KEYWORDS):
            print(f"\n[{i+1}/{len(config.SEARCH_KEYWORDS)}] Scraping: {keyword}")
            query_tweets = scraper.scrape_search_results(keyword, tweets_per_query)
            raw_tweets.extend(query_tweets)
            print(f"  ✓ Collected {len(query_tweets)} tweets from {keyword}")
            logger.info(f"Collected {len(query_tweets)} tweets from {keyword}")
        
        scraper.close()
        
        print(f"\n✓ Collected {len(raw_tweets)} tweets total")
        logger.info(f"Collected {len(raw_tweets)} raw tweets")
        
        if len(raw_tweets) == 0:
            logger.error("No tweets collected. Check connection and Twitter access.")
            return
        
        # Step 2: Data Processing
        print("\n🔧 STEP 2: Data Processing & Normalization")
        print("-" * 70)
        logger.info("Starting data processing phase")
        
        processor = DataProcessor(output_path=config.PROCESSED_DATA_PATH)
        processed_df = processor.process_tweets(raw_tweets)
        
        print(f"✓ Processed {len(processed_df)} tweets")
        print(f"  - Removed duplicates: {len(raw_tweets) - len(processed_df)}")
        logger.info(f"Processed and deduplicated tweets: {len(processed_df)}")
        
        # Generate statistics
        stats = processor.generate_statistics(processed_df)
        processor.export_statistics(stats)
        
        print(f"✓ Unique users: {stats['total_tweets']} from {stats['unique_users']} users")
        
        # Save processed data
        parquet_file = processor.save_to_parquet(processed_df)
        csv_file = processor.save_to_csv(processed_df)
        print(f"✓ Data saved: Parquet & CSV formats")
        
        # Step 3: Signal Generation & Analysis
        print("\n📈 STEP 3: Trading Signal Generation & Analysis")
        print("-" * 70)
        logger.info("Starting signal generation phase")
        
        signal_gen = TradingSignalGenerator(min_confidence=config.MIN_SIGNAL_CONFIDENCE)
        signals_df = signal_gen.generate_signals(processed_df)
        
        print(f"✓ Generated signals for {len(signals_df)} records")
        
        # Get insights
        insight_gen = InsightGenerator()
        market_summary = insight_gen.get_market_summary(signals_df)
        top_signals = insight_gen.get_top_signals(signals_df, top_n=10)
        
        print(f"\nMarket Analysis:")
        # Access nested structure
        overall = market_summary['overall_market']
        print(f"  - Buy Signals: {overall['buy_count']} ({overall['buy_percentage']:.1f}%)")
        print(f"  - Sell Signals: {overall['sell_count']} ({overall['sell_percentage']:.1f}%)")
        print(f"  - Hold Signals: {overall['hold_count']}")
        print(f"  - Market Direction: {overall['market_direction']}")
        print(f"  - Avg Signal Strength: {overall['avg_signal_strength']:.2%}")
        print(f"  - Avg Confidence: {overall['avg_confidence']:.2%}")
        print(f"  - Avg Sentiment: {overall['avg_sentiment']:.2f}")
        
        print(f"\nIndex-Specific Signals:")
        print(f"  Nifty50: {market_summary['nifty50']['signal']} (Sentiment: {market_summary['nifty50']['sentiment']:.2f}, Conviction: {market_summary['nifty50']['conviction']})")
        print(f"  Sensex: {market_summary['sensex']['signal']} (Sentiment: {market_summary['sensex']['sentiment']:.2f}, Conviction: {market_summary['sensex']['conviction']})")
        print(f"  BankNifty: {market_summary['banknifty']['signal']} (Sentiment: {market_summary['banknifty']['sentiment']:.2f}, Conviction: {market_summary['banknifty']['conviction']})")
        
        logger.info(f"Market Summary: {market_summary}")
        
        # Step 4: Visualization
        print("\n📊 STEP 4: Generating Visualizations")
        print("-" * 70)
        logger.info("Starting visualization phase")
        
        visualizer = EfficientVisualizer(output_path="output/visualizations")
        
        viz_files = []
        
        # Generate visualizations
        print("Generating plots...")
        viz1 = visualizer.plot_signal_distribution(signals_df, 
                                                   sample_size=config.VISUALIZATION_SAMPLE_SIZE)
        viz_files.append(viz1)
        print("  ✓ Signal distribution plot")
        
        viz2 = visualizer.plot_engagement_analysis(signals_df,
                                                   sample_size=config.VISUALIZATION_SAMPLE_SIZE)
        viz_files.append(viz2)
        print("  ✓ Engagement analysis plot")
        
        viz3 = visualizer.plot_sentiment_analysis(signals_df,
                                                  sample_size=config.VISUALIZATION_SAMPLE_SIZE)
        viz_files.append(viz3)
        print("  ✓ Sentiment analysis plot")
        
        viz4 = visualizer.plot_temporal_analysis(signals_df)
        if viz4:
            viz_files.append(viz4)
            print("  ✓ Temporal analysis plot")
        
        viz5 = visualizer.plot_hashtag_frequency(signals_df, top_n=20)
        viz_files.append(viz5)
        print("  ✓ Hashtag frequency plot")
        
        # Summary report
        viz6 = visualizer.create_summary_report(signals_df, market_summary, top_signals)
        viz_files.append(viz6)
        print("  ✓ Summary report generated")
        
        print(f"✓ Created {len(viz_files)} visualizations")
        
        # Step 5: Generate Report
        print("\n📋 STEP 5: Generating Final Report")
        print("-" * 70)
        logger.info("Generating final report")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'collection_summary': {
                'total_tweets_collected': len(raw_tweets),
                'tweets_after_processing': len(processed_df),
                'unique_users': stats['unique_users'],
                'keywords_searched': config.SEARCH_KEYWORDS,
            },
            'market_analysis': market_summary,
            'top_signals': top_signals,
            'data_files': {
                'parquet': parquet_file,
                'csv': csv_file,
            },
            'visualizations': viz_files,
            'statistics': {
                'engagement_stats': stats['engagement_stats'],
                'signal_stats': stats['signal_stats'],
                'top_hashtags': stats['hashtags_frequency'],
            }
        }
        
        # Save report
        report_file = Path("output/market_intelligence_report.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        print(f"✓ Report saved to {report_file}")
        logger.info(f"Report saved to {report_file}")
        
        # Print summary
        print("\n" + "="*70)
        print("EXECUTION SUMMARY")
        print("="*70)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Tweets Collected: {len(raw_tweets)}")
        print(f"Tweets Processed: {len(processed_df)}")
        print(f"Unique Users: {stats['unique_users']}")
        print(f"\nMarket Direction: {overall['market_direction']}")
        print(f"Average Sentiment: {overall['avg_sentiment']:.3f}")
        print(f"Average Signal Strength: {overall['avg_signal_strength']:.2%}")
        print(f"\nOutput Files:")
        print(f"  - Parquet: {parquet_file}")
        print(f"  - CSV: {csv_file}")
        print(f"  - Report: {report_file}")
        print(f"  - Visualizations: {len(viz_files)} files generated")
        print("\n" + "="*70)
        
        logger.info("Market Intelligence System execution completed successfully")
        print("\n✅ Execution completed successfully!\n")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user")
        print("\n⚠️ Execution interrupted by user")
        return 1
    
    except Exception as e:
        logger.exception(f"Error during execution: {e}")
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
