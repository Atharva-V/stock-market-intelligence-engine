"""
Twitter/X Scraper Module
Handles data collection from Twitter/X with anti-bot measures and rate limiting
"""

import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json
from urllib.parse import quote

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import requests
from bs4 import BeautifulSoup
from loguru import logger
import pandas as pd


@dataclass
class Tweet:
    """Data class for tweet information"""
    username: str
    timestamp: str
    content: str
    likes: int
    retweets: int
    replies: int
    mentions: List[str]
    hashtags: List[str]
    url: str


class TwitterScraper:
    """
    Scraper for Twitter/X market intelligence data
    Implements anti-bot measures and rate limiting
    """
    
    def __init__(self, headless: bool = True, rate_limit: float = 0.5, email: str = "", password: str = ""):
        """
        Initialize scraper with configuration
        
        Args:
            headless: Run browser in headless mode
            rate_limit: Minimum seconds between requests
            email: Gmail email for Twitter login
            password: Gmail password for Twitter login
        """
        self.headless = headless
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.driver = None
        self.user_agents = self._load_user_agents()
        self.email = email
        self.password = password
        self.logged_in = False
        logger.add("logs/scraper.log", rotation="500 MB", retention="7 days")
        
    def _load_user_agents(self) -> List[str]:
        """Load rotating user agents for anti-bot measures"""
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        ]
        return user_agents
    
    def _apply_rate_limit(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            sleep_time = self.rate_limit - elapsed + random.uniform(0, 0.5)
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _get_random_user_agent(self) -> str:
        """Get random user agent for request"""
        return random.choice(self.user_agents)
    
    def _initialize_driver(self):
        """Initialize Selenium WebDriver with anti-detection measures"""
        try:
            options = webdriver.ChromeOptions()
            
            # Disable headless for Gmail login (needs user interaction sometimes)
            if self.headless:
                options.add_argument("--headless")
            
            # Anti-bot measures
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument(f"user-agent={self._get_random_user_agent()}")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            options.add_argument("--window-size=1920,1080")
            
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=options
            )
            
            # Execute stealth script
            self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': '''
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => false,
                    });
                '''
            })
            
            logger.info("WebDriver initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            raise
    
    def _login_with_gmail(self, email: str, password: str) -> bool:
        """
        Login to Twitter - handles both automated and manual login
        
        Args:
            email: Gmail email address
            password: Gmail password
            
        Returns:
            True if login successful, False otherwise
        """
        try:
            print("[LOGIN] Starting Twitter login...")
            logger.info("Attempting Twitter login")
            
            # Navigate to Twitter
            self.driver.get("https://x.com")
            time.sleep(3)
            
            # Check if already logged in
            try:
                self.driver.find_element(By.CSS_SELECTOR, "[data-testid='SideNav_NewTweet_Button']")
                print("[LOGIN] Already logged in!")
                logger.info("User already logged in")
                return True
            except:
                print("[LOGIN] Not logged in, proceeding with login...")
            
            # Wait for login page and check current URL
            current_url = self.driver.current_url
            page_title = self.driver.title
            print(f"[LOGIN] Current URL: {current_url}")
            print(f"[LOGIN] Page title: {page_title}")
            
            # Simple approach: Try to find email input and enter credentials
            print("[LOGIN] Waiting for login form...")
            
            # Wait longer and be more patient
            max_attempts = 300  # Try for up to 60 seconds
            login_success = False
            
            for attempt in range(max_attempts):
                try:
                    # Check if logged in (home page indicator)
                    self.driver.find_element(By.CSS_SELECTOR, "[data-testid='SideNav_NewTweet_Button']")
                    print("[LOGIN] Success! Home page loaded")
                    login_success = True
                    break
                except:
                    pass
                
                # Try clicking login button if visible
                try:
                    login_buttons = self.driver.find_elements(By.XPATH, "//a[contains(., 'Log in')] | //button[contains(., 'Log in')]")
                    if login_buttons and attempt == 0:
                        login_buttons[0].click()
                        print("[LOGIN] Clicked login button")
                        time.sleep(2)
                        continue
                except:
                    pass
                
                # Try to find email input
                try:
                    email_inputs = self.driver.find_elements(By.XPATH, "//input[@type='email'] | //input[@autocomplete='email'] | //input[@name='text']")
                    if email_inputs and attempt % 5 == 0:
                        print(f"[LOGIN] Found email input, attempting to enter credentials (attempt {attempt+1}/60)...")
                        email_inputs[0].click()
                        time.sleep(0.5)
                        # Clear field first
                        email_inputs[0].send_keys(u'\uffff' + 'a')  
                        email_inputs[0].send_keys('\u0008' * 100)  # Backspace
                        time.sleep(0.3)
                        email_inputs[0].send_keys(email)
                        time.sleep(1)
                        
                        # Look for next button
                        next_buttons = self.driver.find_elements(By.XPATH, "//button[contains(., 'Next')] | //button[@aria-label='Next']")
                        if next_buttons:
                            next_buttons[0].click()
                            print("[LOGIN] Clicked Next button after email")
                            time.sleep(2)
                            continue
                except Exception as e:
                    pass
                
                # Try to find password input
                try:
                    password_fields = self.driver.find_elements(By.XPATH, "//input[@type='password'] | //input[@name='password']")
                    if password_fields and attempt % 5 == 0:
                        print(f"[LOGIN] Found password input, entering password (attempt {attempt+1}/60)...")
                        password_fields[0].click()
                        time.sleep(0.5)
                        password_fields[0].send_keys('\u0008' * 100)  # Backspace
                        time.sleep(0.3)
                        password_fields[0].send_keys(password)
                        time.sleep(1)
                        
                        # Look for login/next button
                        next_buttons = self.driver.find_elements(By.XPATH, "//button[contains(., 'Next')] | //button[contains(., 'Log in')]")
                        if next_buttons:
                            next_buttons[0].click()
                            print("[LOGIN] Clicked login button after password")
                            time.sleep(3)
                            continue
                except Exception as e:
                    pass
                
                time.sleep(1)
            
            if not login_success:
                print("[LOGIN] Timeout waiting for login completion")
                print("[LOGIN] Browser will remain open - you can manually login if needed")
                print("[LOGIN] Waiting 30 more seconds for manual login...")
                
                # Give user 30 seconds to manually login
                for i in range(30):
                    try:
                        self.driver.find_element(By.CSS_SELECTOR, "[data-testid='SideNav_NewTweet_Button']")
                        print("[LOGIN] Manual login detected! Continuing...")
                        return True
                    except:
                        if i % 5 == 0:
                            print(f"[LOGIN] Waiting for manual login... ({300-i}s remaining)")
                        time.sleep(1)
                        continue
                
                logger.warning("Login timeout - manual intervention may be needed")
                return False
            
            return True
                
        except Exception as e:
            print(f"[LOGIN] Login error: {e}")
            logger.error(f"Login failed: {e}")
            return False
    
    def _extract_tweets_from_page(self) -> List[Tweet]:
        """Extract tweets from current page"""
        tweets = []
        
        try:
            # Wait for tweets to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "[data-testid='tweet']"))
            )
            
            tweet_elements = self.driver.find_elements(By.CSS_SELECTOR, "[data-testid='tweet']")
            
            for tweet_elem in tweet_elements:
                try:
                    # Extract username
                    username = tweet_elem.find_element(By.CSS_SELECTOR, "[data-testid='User-Name'] a").get_attribute("href").split('/')[-1]
                    
                    # Extract timestamp
                    timestamp = tweet_elem.find_element(By.CSS_SELECTOR, "time").get_attribute("datetime")
                    
                    # Extract content
                    content = tweet_elem.find_element(By.CSS_SELECTOR, "[data-testid='tweetText']").text
                    
                    # Extract engagement metrics
                    try:
                        likes = int(tweet_elem.find_element(By.CSS_SELECTOR, "[data-testid='like']").text or 0)
                    except:
                        likes = 0
                    
                    try:
                        retweets = int(tweet_elem.find_element(By.CSS_SELECTOR, "[data-testid='retweet']").text or 0)
                    except:
                        retweets = 0
                    
                    try:
                        replies = int(tweet_elem.find_element(By.CSS_SELECTOR, "[data-testid='reply']").text or 0)
                    except:
                        replies = 0
                    
                    # Extract mentions and hashtags
                    mentions = self._extract_mentions(content)
                    hashtags = self._extract_hashtags(content)
                    
                    # Get tweet URL
                    url = tweet_elem.find_element(By.CSS_SELECTOR, "a[href*='/status/']").get_attribute("href")
                    
                    tweet = Tweet(
                        username=username,
                        timestamp=timestamp,
                        content=content,
                        likes=likes,
                        retweets=retweets,
                        replies=replies,
                        mentions=mentions,
                        hashtags=hashtags,
                        url=url
                    )
                    tweets.append(tweet)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract individual tweet: {e}")
                    continue
            
            logger.info(f"Extracted {len(tweets)} tweets from page")
            return tweets
            
        except TimeoutException:
            logger.warning("Timeout waiting for tweets to load")
            return []
        except Exception as e:
            logger.error(f"Error extracting tweets: {e}")
            return []
    
    @staticmethod
    def _extract_mentions(text: str) -> List[str]:
        """Extract @mentions from tweet text"""
        import re
        return re.findall(r'@(\w+)', text)
    
    @staticmethod
    def _extract_hashtags(text: str) -> List[str]:
        """Extract hashtags from tweet text"""
        import re
        return re.findall(r'#(\w+)', text)
    
    def scrape_search_results(self, query: str, max_tweets: int = 100) -> List[Tweet]:
        """
        Scrape tweets for a specific search query
        
        Args:
            query: Search query/hashtag
            max_tweets: Maximum tweets to collect
            
        Returns:
            List of Tweet objects
        """
        if self.driver is None:
            self._initialize_driver()
            
            # Always attempt login (even if no credentials provided - will wait for manual login)
            if not self.logged_in:
                print("[SCRAPER] Initializing Twitter login (manual or automated)...")
                self.logged_in = self._login_with_gmail(self.email, self.password)
                if not self.logged_in:
                    print("[ERROR] Failed to login, scraping may be blocked")
                    logger.error("Login failed, proceeding anyway")
        
        collected_tweets = []
        seen_urls = set()  # Track unique tweet URLs to avoid duplicates
        scroll_pause_time = random.uniform(1.5, 2.5)  # Reduced pause time
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        
        try:
            # Navigate to search with live filter
            # Adding lang:en to get English tweets and India-specific operators
            search_query = f"{query}"
            # Properly URL encode the search query
            encoded_query = quote(search_query)
            search_url = f"https://x.com/search?q={encoded_query}&f=live&lang=en"
            
            logger.info(f"Searching for: {search_url}")
            logger.info(f"Query: {search_query}")
            print(f"[SCRAPER] Loading search results for: {query}")
            print(f"[SCRAPER] Full URL: {search_url}")
            self.driver.get(search_url)
            self._apply_rate_limit()
            
            # WAIT for search results page to fully load before scraping
            print(f"[SCRAPER] Waiting for search results to load...")
            time.sleep(15)  # Give page PLENTY of time to load live feed (increased from 8)
            try:
                WebDriverWait(self.driver, 20).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "[data-testid='tweet']"))
                )
                print(f"[SCRAPER] Search results loaded, starting to scrape...")
            except TimeoutException:
                print(f"[SCRAPER] WARNING: Tweets took too long to load, proceeding anyway...")
                logger.warning("Tweets did not load within timeout")
            
            # Allow more tweets to load while we're not scrolling yet
            print(f"[SCRAPER] Allowing live feed to populate...")
            time.sleep(10)  # Extra wait for more live tweets to load
            
            consecutive_no_new_scrolls = 0
            scroll_count = 0
            
            while len(collected_tweets) < max_tweets:
                scroll_count += 1
                
                # Extract tweets from current scroll position
                new_tweets = self._extract_tweets_from_page()
                
                # Use content-based deduplication
                new_unique_count = 0
                for tweet in new_tweets:
                    if tweet.url not in seen_urls:
                        seen_urls.add(tweet.url)
                        collected_tweets.append(tweet)
                        new_unique_count += 1
                
                # If no new tweets in this scroll, increment counter
                if new_unique_count == 0:
                    consecutive_no_new_scrolls += 1
                    logger.debug(f"Scroll {scroll_count}: No new tweets ({consecutive_no_new_scrolls} consecutive)")
                    # Stop after 50 consecutive scrolls with no new tweets (VERY deep penetration to get 500+)
                    if consecutive_no_new_scrolls >= 50:
                        logger.info(f"Stopping: {consecutive_no_new_scrolls} consecutive scrolls with no new tweets (collected {len(collected_tweets)})")
                        break
                else:
                    consecutive_no_new_scrolls = 0  # Reset counter
                
                if len(collected_tweets) >= max_tweets:
                    break
                
                # Scroll down aggressively (2x viewport height)
                self.driver.execute_script("window.scrollBy(0, 2 * window.innerHeight);")
                # Reduced pause time for faster scraping
                time.sleep(scroll_pause_time * 0.5)
                
                # Don't check page height - Twitter loads dynamically without expanding DOM
                # Only rely on consecutive no-new-tweets counter to detect end
                
        except Exception as e:
            logger.error(f"Error during scraping: {e}")
        
        logger.info(f"Collected {len(collected_tweets)} tweets total in {scroll_count} scrolls")
        return collected_tweets[:max_tweets]
    
    def scrape_multiple_queries(self, queries: List[str], tweets_per_query: int = 500) -> List[Tweet]:
        """
        Scrape tweets for multiple queries
        
        Args:
            queries: List of search queries
            tweets_per_query: Tweets to collect per query
            
        Returns:
            List of all collected tweets
        """
        all_tweets = []
        
        for query in queries:
            try:
                logger.info(f"Scraping query: {query}")
                tweets = self.scrape_search_results(query, tweets_per_query)
                all_tweets.extend(tweets)
                
                # Random delay between queries
                delay = random.uniform(5, 10)
                logger.info(f"Waiting {delay:.1f} seconds before next query")
                time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Error scraping query {query}: {e}")
                continue
        
        return all_tweets
    
    def close(self):
        """Close the WebDriver"""
        if self.driver:
            self.driver.quit()
            logger.info("WebDriver closed")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()


class SharedBrowserScraper:
    """
    Multi-tab Twitter scraper with single authenticated browser
    Logs in ONCE and reuses the session across multiple tabs for parallel scraping
    Much faster than multi-browser approach (no repeated logins)
    """
    
    def __init__(self, num_tabs: int = 4, headless: bool = False, rate_limit: float = 0.5):
        """
        Initialize shared browser scraper
        
        Args:
            num_tabs: Number of concurrent tabs (2-6 recommended, default 4)
            headless: Run browser in headless mode (False recommended for login)
            rate_limit: Rate limit per tab (in seconds)
        """
        self.num_tabs = num_tabs
        self.headless = headless
        self.rate_limit = rate_limit
        self.driver = None
        self.lock = threading.Lock()
        self.all_tweets = []
        self.seen_urls = set()
        self.logged_in = False
        
        logger.info(f"Initializing shared browser scraper with {num_tabs} tabs")
        print(f"[SHARED-BROWSER] Initializing browser with {num_tabs} concurrent tabs...")
    
    def _initialize_browser(self):
        """Initialize single shared browser with login"""
        try:
            print("[SHARED-BROWSER] Starting browser and logging in...")
            scraper = TwitterScraper(
                headless=self.headless,
                rate_limit=self.rate_limit
            )
            scraper._initialize_driver()
            
            # Navigate to Twitter and wait for potential login
            print("[SHARED-BROWSER] Navigating to Twitter/X...")
            scraper.driver.get("https://x.com")
            time.sleep(2)
            
            # Check if we need to login (try to find home timeline)
            max_wait = 60  # Wait up to 60 seconds
            login_detected = False
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                try:
                    # Try to find home page indicator (logged in)
                    scraper.driver.find_element(By.CSS_SELECTOR, "[data-testid='SideNav_NewTweet_Button']")
                    print("[SHARED-BROWSER] ✓ Already logged in or login cached!")
                    login_detected = True
                    break
                except:
                    # Not logged in yet
                    try:
                        # Check if we can find login button (not logged in)
                        scraper.driver.find_element(By.XPATH, "//a[contains(@href, '/login')]")
                        print("[SHARED-BROWSER] ⚠️  Not logged in. Please login manually in the browser window...")
                        print("[SHARED-BROWSER] 📌 You have 60 seconds to complete the login.")
                    except:
                        pass
                    
                    time.sleep(0.5)
            
            if not login_detected:
                print("[SHARED-BROWSER] ⚠️  Login timeout. Attempting to proceed anyway...")
                logger.warning("Login not detected within timeout, proceeding anyway")
            
            self.driver = scraper.driver
            self.base_scraper = scraper
            self.logged_in = True
            
            print("[SHARED-BROWSER] ✓ Browser ready for parallel tab scraping")
            logger.info("Shared browser initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize shared browser: {e}")
            print(f"[SHARED-BROWSER] ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_new_tab(self, tab_id: int) -> str:
        """Open a new tab and return its handle"""
        try:
            self.driver.execute_script("window.open('');")
            handles = self.driver.window_handles
            tab_handle = handles[-1]
            self.driver.switch_to.window(tab_handle)
            
            logger.info(f"Created tab {tab_id}")
            return tab_handle
            
        except Exception as e:
            logger.error(f"Failed to create tab {tab_id}: {e}")
            return None
    
    def _scrape_in_tab(self, tab_handle: str, query: str, max_tweets: int, tab_id: int) -> List[Tweet]:
        """Scrape in a specific tab"""
        try:
            print(f"[TAB {tab_id}] Starting scrape for: {query}")
            
            # Lock when switching tabs to avoid concurrent access issues
            with self.lock:
                # Switch to this tab
                self.driver.switch_to.window(tab_handle)
                
                # Give it a moment to settle
                time.sleep(0.2)
                
                # Use existing scraper's scrape_search_results on this tab
                self.base_scraper.driver = self.driver
            
            # Scrape WITHOUT lock (this is the long operation)
            tweets = self.base_scraper.scrape_search_results(query, max_tweets)
            
            # Thread-safe addition of unique tweets
            with self.lock:
                unique_tweets = []
                for tweet in tweets:
                    if tweet.url not in self.seen_urls:
                        self.seen_urls.add(tweet.url)
                        unique_tweets.append(tweet)
                        self.all_tweets.append(tweet)
                
                print(f"[TAB {tab_id}] Collected {len(unique_tweets)} unique tweets for: {query}")
                logger.info(f"Tab {tab_id}: Collected {len(unique_tweets)} unique tweets for {query}")
            
            return unique_tweets
            
        except Exception as e:
            logger.error(f"Tab {tab_id} error: {e}")
            print(f"[TAB {tab_id}] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def scrape_parallel(self, queries: List[str], tweets_per_query: int = 500) -> List[Tweet]:
        """
        Scrape multiple queries in parallel using browser tabs
        
        Args:
            queries: List of search queries
            tweets_per_query: Tweets to collect per query
            
        Returns:
            List of all unique collected tweets
        """
        if not self._initialize_browser():
            print("❌ Failed to initialize browser. Exiting.")
            return []
        
        print(f"\n{'='*80}")
        print(f"🚀 SHARED-BROWSER PARALLEL SCRAPING")
        print(f"{'='*80}")
        print(f"Queries: {len(queries)}")
        print(f"Target per query: {tweets_per_query}")
        print(f"Concurrent tabs: {min(self.num_tabs, len(queries))}")
        print(f"Estimated max tweets: {len(queries) * tweets_per_query}")
        
        self.all_tweets = []
        self.seen_urls = set()
        
        start_time = time.time()
        tab_handles = {}
        
        try:
            # Create tabs for each query
            for i in range(min(self.num_tabs, len(queries))):
                tab_handle = self._create_new_tab(i)
                if tab_handle:
                    tab_handles[i] = tab_handle
            
            # Scrape queries using thread pool
            with ThreadPoolExecutor(max_workers=self.num_tabs) as executor:
                future_to_query = {}
                for i, query in enumerate(queries):
                    tab_id = i % len(tab_handles)
                    tab_handle = tab_handles[tab_id]
                    future = executor.submit(self._scrape_in_tab, tab_handle, query, tweets_per_query, tab_id)
                    future_to_query[future] = query
                
                # Process completed tasks
                completed = 0
                for future in as_completed(future_to_query):
                    completed += 1
                    query = future_to_query[future]
                    try:
                        result = future.result()
                        print(f"✓ [{completed}/{len(queries)}] Completed: {query} ({len(result)} unique)")
                    except Exception as e:
                        print(f"✗ [{completed}/{len(queries)}] Failed: {query} - {e}")
                        logger.error(f"Task failed for query {query}: {e}")
        
        except Exception as e:
            logger.error(f"Shared browser scraping error: {e}")
            print(f"❌ Error: {e}")
        
        finally:
            self.close()
        
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"📊 RESULTS")
        print(f"{'='*80}")
        print(f"Total Unique Tweets: {len(self.all_tweets)}")
        print(f"Time Elapsed: {elapsed_time:.1f}s")
        print(f"Tweets/Second: {len(self.all_tweets) / elapsed_time:.2f}" if elapsed_time > 0 else "N/A")
        print(f"Avg per Query: {len(self.all_tweets) / len(queries):.0f}" if queries else "N/A")
        
        logger.info(f"Shared browser scraping completed: {len(self.all_tweets)} tweets in {elapsed_time:.1f}s")
        
        return self.all_tweets
    
    def scrape_single_query_fast(self, query: str, max_tweets: int = 2000) -> List[Tweet]:
        """
        Scrape a single query using all available tabs (fastest mode)
        Good for scraping large amounts from one query
        
        Args:
            query: Search query
            max_tweets: Total tweets to collect
            
        Returns:
            List of collected tweets
        """
        if not self._initialize_browser():
            print("❌ Failed to initialize browser. Exiting.")
            return []
        
        tweets_per_tab = max_tweets // self.num_tabs
        
        print(f"\n{'='*80}")
        print(f"🚀 SINGLE QUERY FAST SCRAPING")
        print(f"{'='*80}")
        print(f"Query: {query}")
        print(f"Target: {max_tweets} tweets")
        print(f"Concurrent tabs: {self.num_tabs}")
        print(f"Per tab: {tweets_per_tab} tweets")
        
        self.all_tweets = []
        self.seen_urls = set()
        
        start_time = time.time()
        tab_handles = {}
        
        try:
            # Create tabs
            for i in range(self.num_tabs):
                tab_handle = self._create_new_tab(i)
                if tab_handle:
                    tab_handles[i] = tab_handle
            
            # Scrape in parallel
            with ThreadPoolExecutor(max_workers=self.num_tabs) as executor:
                futures = {
                    executor.submit(self._scrape_in_tab, tab_handles[i], query, tweets_per_tab, i): i
                    for i in tab_handles.keys()
                }
                
                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    tab_id = futures[future]
                    try:
                        result = future.result()
                        print(f"✓ [{completed}/{self.num_tabs}] Tab {tab_id} collected {len(result)} unique tweets")
                    except Exception as e:
                        print(f"✗ [{completed}/{self.num_tabs}] Tab {tab_id} failed: {e}")
                        logger.error(f"Tab {tab_id} failed: {e}")
        
        except Exception as e:
            logger.error(f"Single query scraping error: {e}")
            print(f"❌ Error: {e}")
        
        finally:
            self.close()
        
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"📊 RESULTS")
        print(f"{'='*80}")
        print(f"Total Unique Tweets: {len(self.all_tweets)}")
        print(f"Time Elapsed: {elapsed_time:.1f}s")
        print(f"Tweets/Second: {len(self.all_tweets) / elapsed_time:.2f}" if elapsed_time > 0 else "N/A")
        print(f"Efficiency: {len(self.all_tweets) / max_tweets * 100:.1f}%" if max_tweets > 0 else "N/A")
        
        logger.info(f"Single query scraping completed: {len(self.all_tweets)} tweets in {elapsed_time:.1f}s")
        
        return self.all_tweets
    
    def close(self):
        """Close browser and cleanup"""
        try:
            if self.driver:
                self.driver.quit()
                print("[SHARED-BROWSER] Browser closed")
                logger.info("Shared browser closed")
        except Exception as e:
            logger.error(f"Error closing browser: {e}")


# Backward compatibility: Keep old MultiSessionTwitterScraper name as alias
class MultiSessionTwitterScraper:
    """
    Multi-session Twitter scraper with ONE shared authenticated browser
    Logs in ONCE and runs multiple scraping operations concurrently
    """
    
    def __init__(self, num_sessions: int = 3, headless: bool = False, rate_limit: float = 1.0):
        """
        Initialize multi-session scraper
        
        Args:
            num_sessions: Number of concurrent operations (2-4 recommended)
            headless: Run browser in headless mode
            rate_limit: Rate limit per session (in seconds)
        """
        self.num_sessions = num_sessions
        self.headless = headless
        self.rate_limit = rate_limit
        self.shared_driver = None
        self.lock = threading.Lock()
        self.all_tweets = []
        self.seen_urls = set()
        
        logger.info(f"Initializing multi-session scraper with {num_sessions} sessions")
        print(f"[MULTI-SCRAPER] Initializing {num_sessions} concurrent sessions (shared login)...")
    
    def _initialize_shared_browser(self) -> bool:
        """Initialize single shared browser and login ONCE"""
        try:
            print("[LOGIN] Starting single shared browser...")
            scraper = TwitterScraper(headless=self.headless, rate_limit=self.rate_limit)
            scraper._initialize_driver()
            
            print("[LOGIN] Navigating to Twitter/X...")
            scraper.driver.get("https://x.com")
            time.sleep(2)
            
            # Wait for login or cached session
            print("[LOGIN] Waiting for login (max 60 seconds)...")
            max_wait = 60
            start_time = time.time()
            login_detected = False
            
            while time.time() - start_time < max_wait:
                try:
                    scraper.driver.find_element(By.CSS_SELECTOR, "[data-testid='SideNav_NewTweet_Button']")
                    print("[LOGIN] ✓ Successfully logged in!")
                    login_detected = True
                    break
                except:
                    elapsed = int(time.time() - start_time)
                    print(f"[LOGIN] Waiting... ({elapsed}s/{max_wait}s)", end='\r')
                    time.sleep(0.5)
            
            print()  # New line
            if not login_detected:
                print("[LOGIN] ⚠️  Login timeout")
                scraper.close()
                return False
            
            self.shared_driver = scraper.driver
            self.base_scraper = scraper
            logger.info("Shared browser initialized and logged in")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize shared browser: {e}")
            print(f"[LOGIN] ✗ Error: {e}")
            return False
    
    def _scrape_query_in_shared_browser(self, query: str, max_tweets: int, task_id: int) -> List[Tweet]:
        """Scrape a query using the shared authenticated browser"""
        try:
            print(f"[TASK {task_id}] Starting scrape for: {query}")
            
            # Set the shared driver for this scraping operation
            self.base_scraper.driver = self.shared_driver
            tweets = self.base_scraper.scrape_search_results(query, max_tweets)
            
            # Thread-safe collection of unique tweets
            with self.lock:
                unique_tweets = []
                for tweet in tweets:
                    if tweet.url not in self.seen_urls:
                        self.seen_urls.add(tweet.url)
                        unique_tweets.append(tweet)
                        self.all_tweets.append(tweet)
                
                print(f"[TASK {task_id}] Collected {len(unique_tweets)} unique tweets")
                logger.info(f"Task {task_id}: {len(unique_tweets)} unique tweets for {query}")
            
            return unique_tweets
            
        except Exception as e:
            logger.error(f"Task {task_id} error: {e}")
            print(f"[TASK {task_id}] ERROR: {e}")
            return []
    
    def scrape_queries_parallel(self, queries: List[str], tweets_per_query: int = 500) -> List[Tweet]:
        """
        Scrape multiple queries in parallel using ONE shared authenticated browser
        
        Args:
            queries: List of search queries
            tweets_per_query: Tweets to collect per query
            
        Returns:
            List of all unique collected tweets
        """
        # Initialize shared browser with single login
        if not self._initialize_shared_browser():
            print("❌ Failed to initialize shared browser. Exiting.")
            return []
        
        print(f"\n{'='*80}")
        print(f"🚀 MULTI-SESSION SCRAPING (Shared Authenticated Browser)")
        print(f"{'='*80}")
        print(f"Queries: {len(queries)}")
        print(f"Target per query: {tweets_per_query}")
        print(f"Concurrent tasks: {self.num_sessions}")
        print(f"Estimated max tweets: {len(queries) * tweets_per_query}")
        
        self.all_tweets = []
        self.seen_urls = set()
        
        start_time = time.time()
        
        try:
            with ThreadPoolExecutor(max_workers=self.num_sessions) as executor:
                # Submit all queries
                future_to_query = {}
                for i, query in enumerate(queries):
                    future = executor.submit(
                        self._scrape_query_in_shared_browser,
                        query,
                        tweets_per_query,
                        i
                    )
                    future_to_query[future] = query
                
                # Process results as they complete
                completed = 0
                for future in as_completed(future_to_query):
                    completed += 1
                    query = future_to_query[future]
                    try:
                        result = future.result()
                        print(f"✓ [{completed}/{len(queries)}] Completed: {query} ({len(result)} tweets)")
                    except Exception as e:
                        print(f"✗ [{completed}/{len(queries)}] Failed: {query} - {e}")
                        logger.error(f"Query {query} failed: {e}")
        
        except Exception as e:
            logger.error(f"Scraping error: {e}")
            print(f"❌ Error: {e}")
        
        finally:
            # Close shared browser
            try:
                self.base_scraper.close()
                print("[CLEANUP] Browser closed")
            except:
                pass
        
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"📊 RESULTS")
        print(f"{'='*80}")
        print(f"Total Unique Tweets: {len(self.all_tweets)}")
        print(f"Time Elapsed: {elapsed_time:.1f}s")
        if elapsed_time > 0:
            print(f"Tweets/Second: {len(self.all_tweets) / elapsed_time:.2f}")
        if queries:
            print(f"Avg per Query: {len(self.all_tweets) / len(queries):.0f}")
        
        logger.info(f"Scraping completed: {len(self.all_tweets)} tweets in {elapsed_time:.1f}s")
        
        return self.all_tweets
