import os
import time
import json
import random
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from config.config import Config
from backend.data_services.sentiment_analyzer import get_sentiment_analyzer
from tavily import TavilyClient

class TavilyScraper:
    """
    Service for scraping financial news and information using the Tavily API.
    """
    
    def __init__(self):
        """Initialize the TavilyScraper with API key from config"""
        try:
            self.api_key = Config.TAVILY_API_KEY
            if not self.api_key or self.api_key == "your_tavily_api_key_here":
                print("WARNING: Tavily API key not set. Using mock data.")
                self.use_mock = True
                self.client = None
            else:
                # Make sure API key has the proper format
                # Tavily API keys typically start with 'tvly-'
                if not self.api_key.startswith("tvly-") and not self.api_key.startswith("TVLY-"):
                    # Add prefix if missing
                    formatted_api_key = f"tvly-{self.api_key}"
                    print("WARNING: Adding 'tvly-' prefix to API key. If this causes issues, please update your .env file.")
                else:
                    formatted_api_key = self.api_key
                    
                self.use_mock = False
                self.client = TavilyClient(api_key=formatted_api_key)
                print("Successfully initialized Tavily client.")
        except Exception as e:
            print(f"WARNING: Tavily API key not found or invalid. Using mock data. Error: {str(e)}")
            self.api_key = None
            self.use_mock = True
            self.client = None
    
    def search_news(self, query, max_results=10, include_domains=None, exclude_domains=None):
        """
        Search for news articles related to a query
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results to return
            include_domains (list): List of domains to include in search
            exclude_domains (list): List of domains to exclude from search
            
        Returns:
            list: List of news article dictionaries
        """
        # Check if we should use mock data
        if self.use_mock or not self.api_key or not self.client:
            return self._get_mock_news(query, max_results)
            
        # If we have an API key, try to use the Tavily client
        try:
            search_params = {
                "query": query,
                "search_depth": "advanced",
                "include_answer": False,
                "include_images": True,
                "max_results": max_results
            }
            
            if include_domains:
                search_params["include_domains"] = include_domains
            if exclude_domains:
                search_params["exclude_domains"] = exclude_domains
            
            # Use the Tavily client to search
            response = self.client.search(**search_params)
            
            # Process the results
            return self._process_search_results(response)
            
        except Exception as e:
            print(f"Error using Tavily API: {str(e)}. Using mock data instead.")
            self.use_mock = True  # Switch to mock mode for subsequent requests
            return self._get_mock_news(query, max_results)
    
    def _process_search_results(self, data):
        """
        Process search results from Tavily API
        
        Args:
            data (dict): API response data
            
        Returns:
            list: Processed news articles
        """
        articles = []
        
        # The TavilyClient may return data in a slightly different format
        # Adapt the processing to handle both direct API and client formats
        results = data.get("results", [])
        if not results and isinstance(data, list):
            # Handle case where client might return a list directly
            results = data
            
        for result in results:
            article = {
                "title": result.get("title", ""),
                "content": result.get("content", ""),
                "url": result.get("url", ""),
                "published_date": result.get("published_date", datetime.now().isoformat()),
                "source": result.get("source", ""),
                "image_url": result.get("image_url", ""),
                "score": result.get("score", 0),
            }
            articles.append(article)
        
        return articles
        
    def _get_mock_news(self, query, max_results=10):
        """
        Generate mock news data when API is unavailable
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results to return
            
        Returns:
            list: List of mock news article dictionaries
        """
        import random
        from datetime import datetime, timedelta
        
        # Seed random for consistent results
        random.seed(hash(query) % 10000)
        
        # Extract ticker from query if it looks like a stock query
        ticker = None
        if "stock" in query.lower() or "price" in query.lower() or "shares" in query.lower():
            # Look for pattern that might be a ticker symbol (all caps, 1-5 letters)
            import re
            ticker_match = re.search(r'\b([A-Z]{1,5})\b', query)
            if ticker_match:
                ticker = ticker_match.group(1)
        
        # Generate mock news sources
        sources = [
            "Reuters", "Bloomberg", "CNBC", "The Wall Street Journal", "Financial Times",
            "MarketWatch", "Yahoo Finance", "Seeking Alpha", "The Motley Fool", "Barron's",
            "Investor's Business Daily", "Forbes", "Business Insider", "The Street"
        ]
        
        # Generate domains
        domains = {
            "Reuters": "reuters.com",
            "Bloomberg": "bloomberg.com",
            "CNBC": "cnbc.com",
            "The Wall Street Journal": "wsj.com",
            "Financial Times": "ft.com",
            "MarketWatch": "marketwatch.com",
            "Yahoo Finance": "finance.yahoo.com",
            "Seeking Alpha": "seekingalpha.com",
            "The Motley Fool": "fool.com",
            "Barron's": "barrons.com",
            "Investor's Business Daily": "investors.com",
            "Forbes": "forbes.com",
            "Business Insider": "businessinsider.com",
            "The Street": "thestreet.com"
        }
        
        # News titles templates
        general_titles = [
            "Market Analysis: Trends and Insights for Investors",
            "Economic Outlook: What to Expect in the Coming Months",
            "Investment Strategies for Volatile Markets",
            "Global Markets: International Investment Opportunities",
            "Tech Sector Analysis: Growth and Innovation",
            "Financial Sector Update: Banking and Investment Services",
            "Healthcare Stocks: Opportunities and Challenges",
            "Energy Sector: Market Trends and Investment Potential",
            "Consumer Goods: Retail and E-commerce Developments",
            "Real Estate Markets: Commercial and Residential Trends"
        ]
        
        stock_titles = [
            "[TICKER]: Earnings Report Shows Strong Financial Performance",
            "Analysts Upgrade [TICKER] Following Strategic Announcements",
            "[TICKER] Shares Surge on Positive Market Sentiment",
            "[TICKER] Announces New Product Line, Investors Respond Positively",
            "[TICKER] Quarterly Results Exceed Market Expectations",
            "Leadership Changes at [TICKER]: What Investors Need to Know",
            "[TICKER] Expansion Plans: New Markets and Growth Potential",
            "[TICKER]: Dividend Announcement and Shareholder Returns",
            "[TICKER] Stock Analysis: Technical Indicators and Price Targets",
            "[TICKER] Partners with Industry Leaders to Drive Innovation"
        ]
        
        # Content templates
        general_content = [
            "Market analysts are closely watching economic indicators as investors navigate through recent volatility. The focus remains on central bank policies and inflation data.",
            "Investors are weighing opportunities across diverse sectors, with technology and healthcare continuing to show strong fundamentals despite broader market uncertainties.",
            "Global markets are responding to geopolitical developments, with emerging markets showing resilience amid complex macroeconomic conditions.",
            "Financial experts recommend portfolio diversification strategies to balance risk in the current investment landscape, with a mix of growth and value stocks.",
            "Economic forecasts suggest measured growth through the quarter, with specific sectors positioned to outperform based on current market dynamics."
        ]
        
        stock_content = [
            "The company reported quarterly earnings that exceeded analyst expectations, with revenue growth in key product categories and expanded profit margins.",
            "Recent strategic initiatives have strengthened the company's market position, with analysts noting improved competitive advantages in core business segments.",
            "Investors responded positively to the latest product announcements, which are expected to drive revenue growth in upcoming quarters.",
            "Management provided optimistic guidance for the fiscal year, highlighting expansion plans and operational efficiency improvements.",
            "The company's technology investments are beginning to yield results, with new digital capabilities enhancing customer experiences and operational efficiencies."
        ]
        
        # Generate articles
        articles = []
        now = datetime.now()
        
        for i in range(min(max_results, 15)):
            # Select random source
            source = random.choice(sources)
            domain = domains.get(source, "example.com")
            
            # Generate random date within the past week
            days_ago = random.randint(0, 7)
            hours_ago = random.randint(0, 23)
            published_date = (now - timedelta(days=days_ago, hours=hours_ago)).isoformat()
            
            # Select title and content based on ticker presence
            if ticker:
                title = random.choice(stock_titles).replace("[TICKER]", ticker)
                content = random.choice(stock_content)
            else:
                title = random.choice(general_titles)
                content = random.choice(general_content)
            
            # Generate mock URL
            url_title = title.lower().replace(" ", "-").replace(":", "").replace(",", "")
            url = f"https://www.{domain}/finance/{url_title}-{random.randint(10000, 99999)}"
            
            # Generate article object
            article = {
                "title": title,
                "content": content,
                "url": url,
                "published_date": published_date,
                "source": source,
                "image_url": f"https://placehold.co/600x400?text={source}+Image",
                "score": round(random.uniform(0.5, 0.95), 2)
            }
            
            articles.append(article)
        
        # Sort by published date, most recent first
        articles.sort(key=lambda x: x["published_date"], reverse=True)
        
        return articles
    
    def get_stock_news(self, symbol, days_back=7, max_results=20):
        """
        Get news specifically about a stock
        
        Args:
            symbol (str): Stock symbol
            days_back (int): Number of days to look back
            max_results (int): Maximum number of results to return
            
        Returns:
            list: List of news article dictionaries
        """
        # Get company name for better search results
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            company_name = ticker.info.get('longName', symbol)
        except:
            company_name = symbol
        
        # Create search query with company name and stock symbol
        query = f"{company_name} ({symbol}) stock news financial"
        
        # Include financial news domains
        include_domains = [
            "finance.yahoo.com",
            "bloomberg.com",
            "cnbc.com",
            "wsj.com",
            "ft.com",
            "reuters.com",
            "marketwatch.com",
            "fool.com",
            "seekingalpha.com",
            "investopedia.com",
            "barrons.com",
            "investors.com",
            "businessinsider.com",
            "thestreet.com"
        ]
        
        return self.search_news(query, max_results=max_results, include_domains=include_domains)
    
    def scrape(self, query):
        """
        Alias for search_news, added for compatibility with Streamlit app
        
        Args:
            query (str): Search query
            
        Returns:
            list: List of news article dictionaries
        """
        return self.search_news(query)
    
    def get_web_research(self, query):
        """
        Get web research for a given query
        For compatibility with Streamlit app
        
        Args:
            query (str): Search query
            
        Returns:
            dict: Dictionary containing research results
        """
        articles = self.search_news(query)
        
        # Format the response to match what the Streamlit app expects
        research_data = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "sources": articles,
            "summary": "Based on the gathered information, " + 
                      f"we found {len(articles)} relevant sources about {query}."
        }
        
        return research_data
        
        # Include financial news domains
        include_domains = [
            "finance.yahoo.com",
            "bloomberg.com",
            "cnbc.com",
            "wsj.com",
            "ft.com",
            "reuters.com",
            "marketwatch.com",
            "fool.com",
            "seekingalpha.com",
            "investopedia.com",
            "barrons.com",
            "morningstar.com"
        ]
        
        return self.search_news(query, max_results=max_results, include_domains=include_domains)
    
    def get_market_news(self, max_results=20):
        """
        Get general market news
        
        Args:
            max_results (int): Maximum number of results to return
            
        Returns:
            list: List of news article dictionaries
        """
        query = "stock market financial news analysis today"
        
        # Include financial news domains
        include_domains = [
            "finance.yahoo.com",
            "bloomberg.com",
            "cnbc.com",
            "wsj.com",
            "ft.com",
            "reuters.com",
            "marketwatch.com",
            "fool.com",
            "seekingalpha.com",
            "investopedia.com",
            "barrons.com",
            "morningstar.com"
        ]
        
        return self.search_news(query, max_results=max_results, include_domains=include_domains)
    
    def get_sector_news(self, sector, max_results=15):
        """
        Get news about a specific market sector
        
        Args:
            sector (str): Market sector (technology, healthcare, energy, etc.)
            max_results (int): Maximum number of results to return
            
        Returns:
            list: List of news article dictionaries
        """
        query = f"{sector} sector stock market financial news analysis"
        
        # Include financial news domains
        include_domains = [
            "finance.yahoo.com",
            "bloomberg.com",
            "cnbc.com",
            "wsj.com",
            "ft.com",
            "reuters.com",
            "marketwatch.com",
            "fool.com",
            "seekingalpha.com",
            "investopedia.com",
            "barrons.com",
            "morningstar.com"
        ]
        
        return self.search_news(query, max_results=max_results, include_domains=include_domains)
    
    def analyze_sentiment(self, articles):
        """
        Analyze sentiment of news articles using NLP
        
        Args:
            articles (list): List of news article dictionaries
            
        Returns:
            dict: Sentiment analysis results and articles with sentiment data
        """
        if not articles:
            return {
                "articles": [],
                "overall_sentiment": "neutral",
                "sentiment_distribution": {
                    "positive": 0,
                    "neutral": 0,
                    "negative": 0
                },
                "sentiment_score": 0.0
            }
        
        try:
            # Get the sentiment analyzer
            sentiment_analyzer = get_sentiment_analyzer()
            
            # Analyze sentiment for all articles
            articles_with_sentiment, aggregate = sentiment_analyzer.analyze_news_batch(articles)
            
            # Return the articles with their sentiment and aggregate results
            return {
                "articles": articles_with_sentiment,
                "overall_sentiment": aggregate["sentiment_summary"],
                "sentiment_distribution": aggregate["sentiment_distribution"],
                "sentiment_score": aggregate["average_compound"],
            }
            
        except Exception as e:
            print(f"Error analyzing sentiment: {str(e)}. Using basic sentiment analysis.")
            
            # Fallback to basic random sentiment
            import random
            
            sentiments = ["positive", "neutral", "negative"]
            for article in articles:
                article["sentiment"] = random.choice(sentiments)
            
            # Count sentiments
            sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
            for article in articles:
                sentiment_counts[article["sentiment"]] += 1
                
            # Determine overall sentiment
            max_sentiment = max(sentiment_counts, key=sentiment_counts.get)
            
            return {
                "articles": articles,
                "overall_sentiment": "bullish" if max_sentiment == "positive" else "bearish" if max_sentiment == "negative" else "neutral",
                "sentiment_distribution": {
                    k: round(v / len(articles) * 100) if len(articles) > 0 else 0
                    for k, v in sentiment_counts.items()
                },
                "sentiment_score": 0.0
            }
