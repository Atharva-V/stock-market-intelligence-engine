"""
Test and validation script for Market Intelligence System
Run this to verify all components are working correctly
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all module imports"""
    print("Testing module imports...")
    try:
        from src.scrapers.twitter_scraper import TwitterScraper, AlternativeScraper
        print("  ✓ twitter_scraper imports")
        
        from src.processors.data_processor import DataProcessor
        print("  ✓ data_processor imports")
        
        from src.analyzers.signal_generator import TradingSignalGenerator, SentimentAnalyzer
        print("  ✓ signal_generator imports")
        
        from src.analyzers.visualization import EfficientVisualizer
        print("  ✓ visualization imports")
        
        from src.utils.config import Config, setup_logging
        print("  ✓ config imports")
        
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    try:
        from src.utils.config import Config
        config = Config()
        
        assert config.TARGET_TWEET_COUNT > 0, "Invalid tweet count"
        assert len(config.SEARCH_KEYWORDS) > 0, "No keywords"
        assert config.SCRAPER_RATE_LIMIT > 0, "Invalid rate limit"
        
        print(f"  ✓ Config loaded successfully")
        print(f"    - Target tweets: {config.TARGET_TWEET_COUNT}")
        print(f"    - Keywords: {config.SEARCH_KEYWORDS}")
        print(f"    - Rate limit: {config.SCRAPER_RATE_LIMIT}s")
        
        return True
    except Exception as e:
        print(f"  ✗ Config test failed: {e}")
        return False


def test_sentiment_analyzer():
    """Test sentiment analysis"""
    print("\nTesting sentiment analysis...")
    try:
        from src.analyzers.signal_generator import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer()
        
        # Test positive sentiment
        pos_text = "bullish buy long accumulate green pump"
        pos_score = analyzer.calculate_sentiment_score(pos_text)
        assert pos_score > 0, "Positive sentiment should be > 0"
        
        # Test negative sentiment
        neg_text = "bearish sell short dump red crash"
        neg_score = analyzer.calculate_sentiment_score(neg_text)
        assert neg_score < 0, "Negative sentiment should be < 0"
        
        # Test neutral
        neutral_text = "market price movement trading"
        neutral_score = analyzer.calculate_sentiment_score(neutral_text)
        
        print(f"  ✓ Sentiment analysis working")
        print(f"    - Positive text: {pos_score:.3f}")
        print(f"    - Negative text: {neg_score:.3f}")
        print(f"    - Neutral text: {neutral_score:.3f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Sentiment analysis test failed: {e}")
        return False


def test_data_processing():
    """Test data processing pipeline"""
    print("\nTesting data processor...")
    try:
        from src.processors.data_processor import DataProcessor
        from dataclasses import dataclass
        
        processor = DataProcessor()
        
        # Test text cleaning
        dirty_text = "Check https://t.co/xyz123 out! http://example.com #hashtag @mention"
        clean_text = processor.clean_text(dirty_text)
        
        assert "http" not in clean_text, "URLs should be removed"
        assert len(clean_text) > 0, "Clean text should not be empty"
        
        print(f"  ✓ Data processor working")
        print(f"    - Original: {dirty_text[:50]}...")
        print(f"    - Cleaned: {clean_text[:50]}...")
        
        return True
    except Exception as e:
        print(f"  ✗ Data processor test failed: {e}")
        return False


def test_file_structure():
    """Test that all required files exist"""
    print("\nTesting file structure...")
    try:
        required_files = [
            "main.py",
            "requirements.txt",
            "setup.py",
            ".env.example",
            ".gitignore",
            "README.md",
            "QUICKSTART.md",
            "src/__init__.py",
            "src/scrapers/__init__.py",
            "src/scrapers/twitter_scraper.py",
            "src/processors/__init__.py",
            "src/processors/data_processor.py",
            "src/analyzers/__init__.py",
            "src/analyzers/signal_generator.py",
            "src/analyzers/visualization.py",
            "src/utils/__init__.py",
            "src/utils/config.py",
            ".github/copilot-instructions.md",
        ]
        
        root = Path(__file__).parent
        missing = []
        
        for file in required_files:
            filepath = root / file
            if not filepath.exists():
                missing.append(file)
        
        if missing:
            print(f"  ✗ Missing files: {missing}")
            return False
        
        print(f"  ✓ All {len(required_files)} required files present")
        return True
    except Exception as e:
        print(f"  ✗ File structure test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("MARKET INTELLIGENCE SYSTEM - Verification Tests")
    print("="*70)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_config),
        ("Sentiment Analysis", test_sentiment_analyzer),
        ("Data Processing", test_data_processing),
        ("File Structure", test_file_structure),
    ]
    
    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
    
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:.<40} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*70 + "\n")
    
    if passed == total:
        print("✅ All tests passed! System is ready to run.\n")
        print("Next steps:")
        print("1. Configure .env (optional)")
        print("2. Run: python main.py")
        print("3. Check output/ folder for results")
        return 0
    else:
        print("❌ Some tests failed. Please fix issues before running main.py\n")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
