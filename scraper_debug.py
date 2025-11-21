#!/usr/bin/env python3
"""
Debug script to diagnose Twitter scraper issues
Tests scroll behavior, page loading, memory usage, and tweet extraction
"""

import time
import tracemalloc
from datetime import datetime
from src.scrapers.twitter_scraper import TwitterScraper
from loguru import logger

logger.add("logs/scraper_debug.log", rotation="500 MB", retention="7 days")

def format_bytes(bytes_val):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} TB"

def check_page_status(driver):
    """Check current page state"""
    try:
        # Check scroll position
        scroll_pos = driver.execute_script("return window.scrollY")
        total_height = driver.execute_script("return document.body.scrollHeight")
        viewport_height = driver.execute_script("return window.innerHeight")
        
        print(f"\n📍 PAGE STATUS:")
        print(f"   Scroll Position: {scroll_pos}px")
        print(f"   Total Page Height: {total_height}px")
        print(f"   Viewport Height: {viewport_height}px")
        print(f"   Progress: {scroll_pos / total_height * 100:.1f}% scrolled")
        
        return scroll_pos, total_height, viewport_height
    except Exception as e:
        print(f"❌ Error checking page status: {e}")
        return None, None, None

def count_tweets_on_page(driver):
    """Count visible tweets on current page"""
    try:
        tweets = driver.find_elements("css selector", "[data-testid='tweet']")
        print(f"   Visible Tweets on Page: {len(tweets)}")
        return len(tweets)
    except Exception as e:
        print(f"❌ Error counting tweets: {e}")
        return 0

def check_memory():
    """Check current memory usage"""
    current, peak = tracemalloc.get_traced_memory()
    print(f"\n💾 MEMORY USAGE:")
    print(f"   Current: {format_bytes(current)}")
    print(f"   Peak: {format_bytes(peak)}")

def debug_scrape(query="nifty50", max_tweets=200, max_scrolls=50):
    """Debug scraping session with detailed logging"""
    
    print("\n" + "="*80)
    print("🔍 TWITTER SCRAPER DEBUG SESSION")
    print("="*80)
    print(f"Query: {query}")
    print(f"Target Tweets: {max_tweets}")
    print(f"Max Scrolls: {max_scrolls}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tracemalloc.start()
    
    scraper = TwitterScraper(headless=False, rate_limit=1.0)
    
    try:
        print("\n[1] Initializing driver...")
        scraper._initialize_driver()
        
        print("\n[2] Logging in...")
        login_success = scraper._login_with_gmail("", "")
        scraper.logged_in = login_success
        if not login_success:
            print("⚠️  Login timeout - proceeding with manual login")
        
        print("\n[3] Starting search...")
        from urllib.parse import quote
        encoded_query = quote(query)
        search_url = f"https://x.com/search?q={encoded_query}&f=live&lang=en"
        print(f"   URL: {search_url}")
        scraper.driver.get(search_url)
        time.sleep(5)
        
        print("\n[4] Waiting for initial tweets to load...")
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.common.by import By
        
        try:
            WebDriverWait(scraper.driver, 15).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "[data-testid='tweet']"))
            )
            print("   ✓ Initial tweets loaded")
        except Exception as e:
            print(f"   ⚠️  Timeout waiting for tweets: {e}")
        
        collected_tweets = []
        scroll_count = 0
        consecutive_no_new = 0
        
        print("\n[5] Starting scroll and extraction loop...\n")
        
        initial_scroll_pos, initial_height, _ = check_page_status(scraper.driver)
        initial_tweet_count = count_tweets_on_page(scraper.driver)
        
        while len(collected_tweets) < max_tweets and scroll_count < max_scrolls:
            scroll_count += 1
            
            print(f"\n📜 SCROLL #{scroll_count}")
            print(f"   Collected: {len(collected_tweets)}/{max_tweets}")
            
            # Extract tweets
            new_tweets = scraper._extract_tweets_from_page()
            print(f"   New Tweets Found: {len(new_tweets)}")
            
            # Add to collection (avoid duplicates)
            unique_new = 0
            for tweet in new_tweets:
                if tweet not in collected_tweets:
                    collected_tweets.append(tweet)
                    unique_new += 1
            
            print(f"   Unique Added: {unique_new}")
            
            if unique_new == 0:
                consecutive_no_new += 1
                print(f"   ⚠️  No new tweets ({consecutive_no_new} consecutive)")
            else:
                consecutive_no_new = 0
            
            # Check page status
            check_page_status(scraper.driver)
            count_tweets_on_page(scraper.driver)
            
            # Scroll down
            print(f"   Scrolling down...")
            scroll_before = scraper.driver.execute_script("return window.scrollY")
            scraper.driver.execute_script("window.scrollBy(0, window.innerHeight);")
            time.sleep(2)  # Wait for content to load
            scroll_after = scraper.driver.execute_script("return window.scrollY")
            scroll_delta = scroll_after - scroll_before
            
            print(f"   Scroll Delta: {scroll_delta}px (before: {scroll_before}, after: {scroll_after})")
            
            if scroll_delta == 0:
                print(f"   ⚠️  NO SCROLL - Possible end of page or infinite scroll issue")
                if consecutive_no_new >= 2:
                    print(f"   🛑 STOP: No scrolling + no new tweets x2")
                    break
            
            # Check memory
            check_memory()
            
            # Safety: 3 consecutive scroll failures = stop
            if consecutive_no_new >= 3:
                print(f"\n🛑 STOPPING: 3 consecutive scrolls with no new tweets")
                break
            
            # Small pause between scrolls
            time.sleep(1)
        
        print("\n" + "="*80)
        print(f"✅ SCRAPING COMPLETE")
        print(f"   Total Tweets Collected: {len(collected_tweets)}")
        print(f"   Total Scrolls: {scroll_count}")
        print(f"   Scrolls/Tweet Ratio: {scroll_count / len(collected_tweets) if collected_tweets else 'N/A'}")
        print(f"   Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        check_memory()
        
        # Summary statistics
        print(f"\n📊 SUMMARY:")
        print(f"   Target: {max_tweets}")
        print(f"   Achieved: {len(collected_tweets)} ({len(collected_tweets)/max_tweets*100:.1f}%)")
        
        if len(collected_tweets) > 0:
            print(f"\n📝 SAMPLE TWEETS:")
            for i, tweet in enumerate(collected_tweets[:3], 1):
                print(f"   {i}. @{tweet.username}: {tweet.content[:60]}...")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(traceback.format_exc())
        logger.error(f"Debug scrape failed: {e}\n{traceback.format_exc()}")
    
    finally:
        print("\n[6] Closing driver...")
        scraper.close()
        tracemalloc.stop()
        print("✓ Debug session ended")

if __name__ == "__main__":
    import traceback
    debug_scrape(query="nifty50", max_tweets=200, max_scrolls=50)
