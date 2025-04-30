import json
import statistics
import time
import re
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from backend.agent_system.agents.base_agent import BaseAgent
from backend.agent_system.event_bus import EventBus, Event, EventType
from backend.data_services.tavily_scraper import TavilyScraper
from backend.data_services.supabase_client import SupabaseClient
from config.config import Config

# Import OpenAI for AI-enhanced analysis
import openai
from openai import OpenAI

class NewsSentimentAgent(BaseAgent):
    """
    Agent responsible for analyzing news sentiment.
    Collects news articles and performs sentiment analysis to gauge market sentiment.
    """
    
    def __init__(self, polling_interval: int = 3600):
        """
        Initialize the News Sentiment Agent
        
        Args:
            polling_interval (int): Interval between agent runs in seconds
        """
        super().__init__(name="NewsSentimentAgent", polling_interval=polling_interval)
        
        # Initialize logger
        self.logger.info("Initializing News Sentiment Agent")
        
        # Initialize services
        self.tavily_scraper = TavilyScraper()
        self.supabase_client = SupabaseClient()
        
        # Initialize sentiment database
        self.sentiment_db = {}
        
        # Initialize OpenAI if API key is available
        try:
            self.openai_api_key = Config.OPENAI_API_KEY
            if self.openai_api_key and self.openai_api_key != "your_openai_api_key_here":
                # Initialize OpenAI client for v1.0.0+ SDK
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                self.use_openai = True
                self.logger.info("OpenAI API key found, will use for enhanced sentiment analysis")
            else:
                self.use_openai = False
                self.logger.info("OpenAI API key not found, using keyword-based sentiment analysis only")
        except Exception as e:
            self.use_openai = False
            self.logger.warning(f"Error setting up OpenAI: {e}, using keyword-based sentiment analysis only")
    
    def run(self):
        """
        Main agent polling function to periodically analyze market sentiment
        """
        self.logger.info("Running News Sentiment Agent polling function")
        
        try:
            # Analyze market sentiment
            self.analyze_market_sentiment()
            
            # Sleep until next polling interval
            self.logger.info(f"Sleeping for {self.polling_interval} seconds")
            time.sleep(self.polling_interval)
            
        except Exception as e:
            self.logger.error(f"Error in News Sentiment Agent polling function: {e}")
            raise
    
    def analyze_market_sentiment(self):
        """
        Analyze market sentiment based on market-wide news
        """
        self.logger.info("Analyzing market news sentiment")
        
        try:
            # Get market news from Tavily
            market_query = "stock market news financial economy"
            news_items = self.tavily_scraper.search_news(market_query, max_results=20)
            
            # Filter relevant news
            news_items = self._filter_financial_news(news_items)
            
            # Store news in database
            if news_items:
                self.supabase_client.store_news_articles(news_items, category="market")
            
            # Analyze sentiment
            sentiment_results = self._analyze_sentiment(news_items, entity_type="market", entity_id="general")
            
            # Store analysis results
            self._store_analysis_results("MARKET", sentiment_results)
            
            # Publish analysis completed event
            self._publish_event(
                event_type=EventType.SENTIMENT_ANALYSIS_COMPLETED,
                data={
                    'symbol': "MARKET",
                    'analysis': sentiment_results
                }
            )
            
            self.logger.info("Completed market news sentiment analysis")
            
        except Exception as e:
            self.logger.error(f"Error in market news sentiment analysis: {e}")
            raise
    
    def _analyze_sentiment(self, news_items: List[Dict[str, Any]], entity_type: str, entity_id: str) -> Dict[str, Any]:
        """
        Analyze sentiment of news articles
        
        Args:
            news_items (List[Dict[str, Any]]): List of news articles
            entity_type (str): Type of entity (stock, market, sector)
            entity_id (str): ID of the entity (symbol, "general", sector name)
            
        Returns:
            Dict[str, Any]: Sentiment analysis results
        """
        # In a production system, this would use OpenAI API or another NLP service
        
        if not news_items:
            return {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "date": datetime.now().isoformat(),
                "overall_sentiment": "neutral",
                "sentiment_score": 50,  # Use 50 for neutral on 0-100 scale
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "total_articles": 0,
                "top_positive_headlines": [],
                "top_negative_headlines": [],
                "key_topics": [],
                "analysis_type": "news_sentiment"
            }
        
        # Enhanced keyword-based sentiment analysis with weighted keywords
        # Use weighted keywords for better sentiment calculation
        positive_keywords = {
            'earnings_beat': 1.5, 'upgrade': 1.3, 'growth': 1.2,
            'expansion': 1.2, 'partnership': 1.1, 'innovation': 1.2,
            'outperform': 1.3, 'record': 1.2, 'acquisition': 1.1,
            'profit': 1.2, 'surge': 1.2, 'rise': 1.1, 'gain': 1.1, 
            'positive': 1.1, 'up': 0.9, 'bullish': 1.3, 'beat': 1.2, 
            'exceed': 1.2, 'strong': 1.1, 'success': 1.2, 'opportunity': 1.0,
            'dividend': 1.4, 'buyback': 1.3
        }
        
        negative_keywords = {
            'downgrade': -1.3, 'miss': -1.2, 'decline': -1.2,
            'lawsuit': -1.4, 'investigation': -1.3, 'bearish': -1.2,
            'recession': -1.3, 'bankruptcy': -1.5, 'restructuring': -1.1,
            'loss': -1.2, 'drop': -1.1, 'fall': -1.1, 'down': -0.9, 
            'negative': -1.1, 'underperform': -1.2, 'weak': -1.1, 
            'struggle': -1.1, 'concern': -1.0, 'risk': -0.9, 'warning': -1.1,
            'debt': -1.3, 'layoff': -1.4, 'cut': -1.1, 'reduce': -1.0
        }
        
        # Analyze each article with weighted keyword approach and time decay
        article_sentiments = []
        positive_articles = []
        negative_articles = []
        neutral_articles = []
        all_words = []
        now = datetime.now()
        
        # Track risk keywords for reporting like the risk assessment agent
        risk_keywords_found = {
            "high_risk": [],
            "moderate_risk": [],
            "positive": []
        }
        
        for article in news_items:
            title = article.get("title", "")
            content = article.get("content", article.get("description", ""))
            combined_text = f"{title} {content}".lower()
            
            # Calculate keyword-based sentiment using weighted approach
            keyword_sentiment = 0
            
            # Check positive keywords with regex pattern matching
            for keyword, weight in positive_keywords.items():
                if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', combined_text):
                    keyword_sentiment += weight
                    # Track for risk reporting
                    if keyword not in risk_keywords_found["positive"]:
                        risk_keywords_found["positive"].append(keyword)
                    
            # Check negative keywords with regex pattern matching
            for keyword, weight in negative_keywords.items():
                if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', combined_text):
                    keyword_sentiment += weight
                    # Track for risk reporting based on severity
                    if weight < -1.2:
                        if keyword not in risk_keywords_found["high_risk"]:
                            risk_keywords_found["high_risk"].append(keyword)
                    else:
                        if keyword not in risk_keywords_found["moderate_risk"]:
                            risk_keywords_found["moderate_risk"].append(keyword)
            
            # Normalize sentiment score to [-1, 1] range
            max_possible = max(3.0, abs(keyword_sentiment))  # Prevent division by zero
            sentiment_score = keyword_sentiment / max_possible
            sentiment_score = max(min(sentiment_score, 1.0), -1.0)  # Clamp to [-1, 1]
            
            # Apply time decay if date is available
            try:
                news_date_str = article.get("date", article.get("published_at", now.isoformat()))
                news_date = datetime.fromisoformat(news_date_str.split("T")[0])
                days_old = (now - news_date).days
                time_factor = max(0.5, 1.0 - (days_old * 0.1))  # Newer articles have more weight
                sentiment_score *= time_factor
            except:
                pass  # If date parsing fails, don't apply time decay
            
            # Generate a better summary using Tavily's content or OpenAI if available
            summary = ""
            if self.use_openai and content and len(content) > 100:
                try:
                    # Use OpenAI to generate a concise summary using v1.0.0+ SDK
                    response = self.openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that summarizes financial news articles concisely."},
                            {"role": "user", "content": f"Please summarize this financial news article in 2-3 sentences: {title}\n\n{content[:1000]}"}
                        ],
                        max_tokens=150,
                        temperature=0.3
                    )
                    if response.choices and response.choices[0].message.content:
                        summary = response.choices[0].message.content.strip()
                    else:
                        # Fallback to simple summary
                        summary = content[:200] + "..." if len(content) > 200 else content
                except Exception as e:
                    self.logger.warning(f"Error generating OpenAI summary: {e}")
                    # Fallback to simple summary
                    summary = content[:200] + "..." if len(content) > 200 else content
            elif content:
                # Use Tavily content for summary (first 200 chars)
                summary = content[:200] + "..." if len(content) > 200 else content
            elif title:
                summary = f"Article about {title}"
            
            # Determine sentiment category
            if sentiment_score > 0.2:
                sentiment = "positive"
                positive_articles.append(article)
            elif sentiment_score < -0.2:
                sentiment = "negative"
                negative_articles.append(article)
            else:
                sentiment = "neutral"
                neutral_articles.append(article)
            
            # Add to article sentiments with summary
            article_sentiments.append({
                "title": title,
                "sentiment": sentiment,
                "sentiment_score": sentiment_score,
                "url": article.get("url", ""),
                "date": article.get("date", ""),
                "source": article.get("source", "Financial News"),
                "summary": summary
            })
            
            # Add words for topic extraction
            words = combined_text.split()
            all_words.extend(words)
        
        # Calculate overall sentiment with weighted approach
        positive_count = len(positive_articles)
        negative_count = len(negative_articles)
        neutral_count = len(neutral_articles)
        total_articles = len(news_items)
        
        # Determine overall sentiment category
        if positive_count > negative_count and positive_count > neutral_count:
            overall_sentiment = "positive"
        elif negative_count > positive_count and negative_count > neutral_count:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"
        
        # Calculate overall sentiment score normalized to 0-100 scale like risk assessment
        sentiment_scores = [article["sentiment_score"] for article in article_sentiments]
        sorted_sentiments = sorted(article_sentiments, key=lambda x: x.get("date", ""), reverse=True)
        
        if sentiment_scores:
            # Calculate weighted scores based on recency
            weighted_scores = []
            recent_cutoff = now - timedelta(days=7)
            
            for article in article_sentiments:
                try:
                    article_date_str = article.get("date", now.isoformat())
                    article_date = datetime.fromisoformat(article_date_str.split("T")[0])
                    days_old = max(0, (now - article_date).days)
                    # More recent articles have more weight
                    article_weight = max(0.5, 1.0 - (days_old * 0.05))
                    weighted_scores.append(article["sentiment_score"] * article_weight)
                except:
                    weighted_scores.append(article["sentiment_score"])
            
            # Calculate overall score and normalize to 0-100 scale like risk assessment
            overall_score = statistics.mean(weighted_scores)
            sentiment_score = (overall_score + 1) * 50  # Convert from [-1,1] to [0,100]
            
            # Adjust based on risk keywords like risk assessment
            high_risk_penalty = len(risk_keywords_found["high_risk"]) * 3
            moderate_risk_penalty = len(risk_keywords_found["moderate_risk"]) * 1.5
            positive_bonus = len(risk_keywords_found["positive"]) * 2
            
            # Final risk-adjusted score
            sentiment_score = max(0, min(100, sentiment_score - high_risk_penalty - moderate_risk_penalty + positive_bonus))
        else:
            sentiment_score = 50  # Neutral if no articles
            overall_score = 0
        
        # Get top headlines
        top_positive_headlines = [
            {"title": a.get("title", ""), "url": a.get("url", "")} 
            for a in sorted(positive_articles, key=lambda x: x.get("title", ""), reverse=True)[:3]
        ]
        
        top_negative_headlines = [
            {"title": a.get("title", ""), "url": a.get("url", "")} 
            for a in sorted(negative_articles, key=lambda x: x.get("title", ""), reverse=True)[:3]
        ]
        
        # Extract key topics (word frequency)
        word_counts = {}
        stopwords = ["the", "and", "a", "to", "of", "in", "is", "that", "it", "with", "for", "as", "on", "by", "at"]
        
        for word in all_words:
            if len(word) > 3 and word not in stopwords:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        key_topics = [
            word for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        # Generate a meaningful analysis summary
        if entity_type == "stock":
            symbol = entity_id.upper()
            if overall_sentiment == "positive":
                analysis = f"News coverage for {symbol} appears primarily positive with {positive_count} positive headlines vs {negative_count} negative."
            elif overall_sentiment == "negative":
                analysis = f"News coverage for {symbol} shows negative sentiment with {negative_count} negative headlines vs {positive_count} positive."
            else:
                analysis = f"News coverage for {symbol} appears neutral to mixed with {positive_count} positive, {negative_count} negative, and {neutral_count} neutral headlines."
                
            # Add key topics to analysis
            if key_topics:
                topics_str = ", ".join(key_topics[:5])
                analysis += f" Key topics include: {topics_str}."
        else:
            analysis = f"Overall {entity_type} sentiment appears {overall_sentiment}."
        
        # Get the most significant headline
        max_abs_score = 0
        significant_headline = "No significant headlines found"
        for a in article_sentiments:
            score = abs(a.get("sentiment_score", 0))
            if score > max_abs_score:
                max_abs_score = score
                significant_headline = a.get("title", "")
        
        # Calculate sentiment trend (compare recent vs older articles)
        sentiment_trend = 0
        if len(sorted_sentiments) >= 4:
            recent_half = sorted_sentiments[:len(sorted_sentiments)//2]
            older_half = sorted_sentiments[len(sorted_sentiments)//2:]
            recent_sentiment = sum(a.get("sentiment_score", 0) for a in recent_half) / len(recent_half)
            older_sentiment = sum(a.get("sentiment_score", 0) for a in older_half) / len(older_half)
            sentiment_trend = recent_sentiment - older_sentiment
        
        # Compile enhanced sentiment analysis results
        sentiment_results = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "date": datetime.now().isoformat(),
            "overall_sentiment": overall_sentiment,
            "sentiment_score": sentiment_score,  # On 0-100 scale like risk assessment
            "score": sentiment_score / 100,  # Add 0-1 scale for backward compatibility
            "sentiment_trend": round(sentiment_trend, 2),
            "sentiment_label": overall_sentiment,
            "headline": significant_headline,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "total_articles": total_articles,
            "article_count": total_articles,
            "sentiment_breakdown": {
                "positive": (positive_count / total_articles) * 100 if total_articles > 0 else 0,
                "neutral": (neutral_count / total_articles) * 100 if total_articles > 0 else 0,
                "negative": (negative_count / total_articles) * 100 if total_articles > 0 else 0
            },
            "breakdown": {  # Add explicit breakdown field for UI compatibility
                "positive": positive_count,  # Use actual counts instead of percentages
                "neutral": neutral_count,
                "negative": negative_count
            },
            "risk_keywords": {  # Add risk keywords like risk assessment
                "high_risk": risk_keywords_found["high_risk"],
                "moderate_risk": risk_keywords_found["moderate_risk"],
                "positive": risk_keywords_found["positive"]
            },
            "news_risk_score": round(sentiment_score, 2),  # Same format as risk assessment
            "top_positive_headlines": top_positive_headlines,
            "top_negative_headlines": top_negative_headlines,
            "key_topics": key_topics,
            "analysis": analysis,
            "summary": analysis,  # Add summary field for UI compatibility
            "articles": article_sentiments,  # Add full article data for detailed display
            "recent_articles": sorted_sentiments[:5],  # Add recent articles for trending display
            "recent_headlines": [a.get("title", "") for a in sorted_sentiments[:3]],  # For simple display
            "analysis_type": "news_sentiment"
        }
        
        return sentiment_results
    
    def _filter_financial_news(self, news_items: List[Dict[str, Any]], days: int = 7) -> List[Dict[str, Any]]:
        """
        Filter news items to only include financial news from the past X days
        
        Args:
            news_items (List[Dict[str, Any]]): List of news articles
            days (int): Number of days to include
            
        Returns:
            List[Dict[str, Any]]: Filtered news items
        """
        financial_domains = [
            "finance.yahoo.com",
            "seekingalpha.com",
            "fool.com",
            "cnbc.com",
            "bloomberg.com",
            "reuters.com",
            "marketwatch.com",
            "investors.com"
        ]
        
        # Filter by date if needed
        cutoff_date = datetime.now() - timedelta(days=days)
        filtered_news = []
        
        for article in news_items:
            try:
                # Extract date in various formats
                date_str = article.get("published_date", article.get("date", ""))
                if date_str:
                    # Handle different date formats
                    if "T" in date_str:
                        date_str = date_str.split("T")[0]
                    article_date = datetime.fromisoformat(date_str)
                    
                    # Keep if article is within timeframe
                    if article_date >= cutoff_date:
                        filtered_news.append(article)
                else:
                    # If no date, include it anyway
                    filtered_news.append(article)
            except Exception:
                # If date parsing fails, include the article
                filtered_news.append(article)
                
        return filtered_news
    
    def get_news_sentiment(self, symbol: str = None) -> Optional[Dict[str, Any]]:
        """
        Get the latest news sentiment analysis for a symbol or market
        
        Args:
            symbol (str, optional): Stock symbol or None for market sentiment
            
        Returns:
            Optional[Dict[str, Any]]: Latest sentiment analysis or None if not found
        """
        try:
            if symbol:
                # For specific stock, first try to get news from Tavily
                news_items = self.tavily_scraper.search_news(f"{symbol} stock news", max_results=15)
                
                # Filter and analyze sentiment
                if news_items:
                    news_items = self._filter_financial_news(news_items)
                    
                    # Analyze sentiment with the fresh news data
                    sentiment_data = self._analyze_sentiment(news_items, "stock", symbol)
                    
                    # Optionally store in database
                    self._store_analysis_results(symbol, sentiment_data)
                    
                    # Return the fresh analysis
                    return sentiment_data
                
                # If no news or Tavily fails, try to get from database
                symbol_to_query = symbol
            else:
                # For market sentiment, just query the database
                symbol_to_query = "MARKET"
                
            # Try to get the latest sentiment from the database
            results = self.supabase_client.get_analysis_results(
                symbol=symbol_to_query,
                analysis_type="news_sentiment",
                limit=1
            )
            
            if not results.empty:
                result_dict = results.iloc[0].to_dict()
                
                # Convert string representations back to objects
                for key in ["top_positive_headlines", "top_negative_headlines", "articles"]:
                    if key in result_dict and isinstance(result_dict[key], str):
                        result_dict[key] = json.loads(result_dict[key])
                
                return result_dict
                
            # If nothing in database, analyze with mock data
            if symbol:
                # For specific stock, get basic news through fallback methods
                from backend.data_services.stock_client import StockClient
                stock_client = StockClient()
                news_data = stock_client.get_news(symbol)
                
                if news_data and "data" in news_data and news_data["data"]:
                    news_items = news_data["data"]
                    return self._analyze_sentiment(news_items, "stock", symbol)
            
            # Final fallback - return basic sentiment
            return {
                "entity_type": "stock" if symbol else "market",
                "entity_id": symbol if symbol else "general",
                "date": datetime.now().isoformat(),
                "overall_sentiment": "neutral",
                "sentiment_score": 50,
                "score": 0.5,
                "sentiment_trend": 0,
                "sentiment_label": "neutral",
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "total_articles": 0,
                "sentiment_breakdown": {"positive": 33.3, "neutral": 33.4, "negative": 33.3},
                "breakdown": {"positive": 33.3, "neutral": 33.4, "negative": 33.3},
                "risk_keywords": {"high_risk": [], "moderate_risk": [], "positive": []},
                "summary": f"No news sentiment data available for {symbol if symbol else 'market'}.",
                "analysis": f"No news sentiment data available for {symbol if symbol else 'market'}."
            }
                
        except Exception as e:
            self.logger.error(f"Error getting news sentiment for {symbol}: {e}")
            return None
    
    def _store_analysis_results(self, symbol: str, analysis: Dict[str, Any]):
        """
        Store sentiment analysis results in the database
        
        Args:
            symbol (str): Stock symbol
            analysis (Dict[str, Any]): Sentiment analysis results
        """
        try:
            # Convert complex objects to strings
            analysis_copy = analysis.copy()
            for key in ["top_positive_headlines", "top_negative_headlines", "articles"]:
                if key in analysis_copy:
                    analysis_copy[key] = json.dumps(analysis_copy[key])
            
            # Store in database - using the correct parameter names
            self.supabase_client.store_analysis_results(
                symbol=symbol,
                analysis_type="news_sentiment",
                data=analysis_copy  # Changed to the correct parameter name 'data'
            )
            
            self.logger.info(f"Stored news sentiment analysis for {symbol}")
        except Exception as e:
            self.logger.error(f"Error storing news sentiment analysis: {e}")
    
    def _publish_event(self, event_type: EventType, data: Dict[str, Any]):
        """
        Publish an event to the event bus
        
        Args:
            event_type (EventType): Type of event
            data (Dict[str, Any]): Event data
        """
        event = Event(
            type=event_type,
            data=data,
            source=self.name
        )
        EventBus.publish(event)
