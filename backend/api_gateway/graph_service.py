import logging
from typing import Dict, Any, List, Optional, Union
import json
from datetime import datetime, timedelta

from backend.agent_system.event_bus import EventBus, Event, EventType
from backend.data_services.stock_server import StockServer
from backend.data_services.tavily_scraper import TavilyScraper
from backend.data_services.supabase_client import SupabaseClient
from backend.agent_system.agents.market_position_agent import MarketPositionAgent
from backend.agent_system.agents.news_sentiment_agent import NewsSentimentAgent
from backend.agent_system.agents.institutional_activity_agent import InstitutionalActivityAgent
from backend.agent_system.agents.risk_assessment_agent import RiskAssessmentAgent
from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GraphService")

class GraphService:
    """
    API Gateway service that handles requests and coordinates between frontend and backend.
    Implements a graph-like structure for data relationships and dependencies.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one GraphService instance exists"""
        if cls._instance is None:
            cls._instance = super(GraphService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the GraphService (only once due to singleton pattern)"""
        if self._initialized:
            return
            
        # Initialize services
        self.stock_server = StockServer()
        self.tavily_scraper = TavilyScraper()
        self.supabase_client = SupabaseClient()
        self.event_bus = EventBus()
        
        # Initialize agents
        self.market_position_agent = MarketPositionAgent()
        self.news_sentiment_agent = NewsSentimentAgent()
        self.institutional_activity_agent = InstitutionalActivityAgent()
        self.risk_assessment_agent = RiskAssessmentAgent()
        
        # Start event bus
        self.event_bus.start()
        
        # Start agents
        self._start_agents()
        
        self._initialized = True
        logger.info("GraphService initialized")
    
    def _start_agents(self):
        """Start all agents"""
        self.market_position_agent.start()
        self.news_sentiment_agent.start()
        self.institutional_activity_agent.start()
        self.risk_assessment_agent.start()
        logger.info("All agents started")
    
    def _stop_agents(self):
        """Stop all agents"""
        self.market_position_agent.stop()
        self.news_sentiment_agent.stop()
        self.institutional_activity_agent.stop()
        self.risk_assessment_agent.stop()
        logger.info("All agents stopped")
    
    def shutdown(self):
        """Shutdown the service"""
        self._stop_agents()
        self.event_bus.stop()
        logger.info("GraphService shutdown")
    
    def get_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> Dict[str, Any]:
        """
        Get stock data for a symbol
        
        Args:
            symbol (str): Stock symbol
            period (str): Time period
            interval (str): Data interval
            
        Returns:
            Dict[str, Any]: Stock data
        """
        try:
            data = self.stock_server.get_historical_data(symbol, period=period, interval=interval)
            
            # Calculate technical indicators
            data_with_indicators = self.stock_server.calculate_technical_indicators(data)
            
            # Convert to dict for JSON serialization
            result = {
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "data": data_with_indicators.to_dict(orient="records")
            }
            
            return result
        except Exception as e:
            logger.error(f"Error getting stock data for {symbol}: {e}")
            return {"error": str(e)}
    
    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get company information for a symbol
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict[str, Any]: Company information
        """
        try:
            info = self.stock_server.get_company_overview(symbol)
            
            return {
                "symbol": symbol,
                "company_info": info
            }
        except Exception as e:
            logger.error(f"Error getting company info for {symbol}: {e}")
            return {"error": str(e)}
    
    def get_news(self, symbol: str = None, max_results: int = 20) -> Dict[str, Any]:
        """
        Get news for a symbol or general market news
        
        Args:
            symbol (str, optional): Stock symbol or None for market news
            max_results (int): Maximum number of results
            
        Returns:
            Dict[str, Any]: News data
        """
        try:
            if symbol:
                news_items = self.tavily_scraper.get_stock_news(symbol, max_results=max_results)
                title = f"News for {symbol}"
            else:
                news_items = self.tavily_scraper.get_market_news(max_results=max_results)
                title = "Market News"
            
            return {
                "title": title,
                "news_items": news_items
            }
        except Exception as e:
            logger.error(f"Error getting news: {e}")
            return {"error": str(e)}
    
    def get_market_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get market position analysis for a symbol
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict[str, Any]: Market position analysis
        """
        try:
            # Try to get existing analysis
            analysis = self.market_position_agent.get_market_position(symbol)
            
            if analysis:
                # Check if analysis is recent (within 24 hours)
                analysis_date = datetime.fromisoformat(analysis.get("date", ""))
                if datetime.now() - analysis_date < timedelta(hours=24):
                    return {"symbol": symbol, "analysis": analysis}
            
            # Trigger new analysis if none exists or is outdated
            self._publish_user_request("market_position", "analyze_symbol", {"symbol": symbol})
            
            # Return message that analysis is being generated
            return {
                "symbol": symbol,
                "status": "generating",
                "message": "Market position analysis is being generated. Please check back in a few moments."
            }
        except Exception as e:
            logger.error(f"Error getting market position for {symbol}: {e}")
            return {"error": str(e)}
    
    def get_news_sentiment(self, symbol: str = None) -> Dict[str, Any]:
        """
        Get news sentiment analysis for a symbol or market
        
        Args:
            symbol (str, optional): Stock symbol or None for market sentiment
            
        Returns:
            Dict[str, Any]: News sentiment analysis
        """
        try:
            # Try to get existing analysis
            analysis = self.news_sentiment_agent.get_news_sentiment(symbol)
            
            if analysis:
                # Check if analysis is recent (within 24 hours)
                analysis_date = datetime.fromisoformat(analysis.get("date", ""))
                if datetime.now() - analysis_date < timedelta(hours=24):
                    return {"symbol": symbol or "MARKET", "analysis": analysis}
            
            # Trigger new analysis if none exists or is outdated
            if symbol:
                self._publish_user_request("news_sentiment", "analyze_news_for_symbol", {"symbol": symbol})
            else:
                self._publish_user_request("news_sentiment", "analyze_market_news", {})
            
            # Return message that analysis is being generated
            return {
                "symbol": symbol or "MARKET",
                "status": "generating",
                "message": "News sentiment analysis is being generated. Please check back in a few moments."
            }
        except Exception as e:
            logger.error(f"Error getting news sentiment for {symbol or 'MARKET'}: {e}")
            return {"error": str(e)}
    
    def get_institutional_activity(self, symbol: str) -> Dict[str, Any]:
        """
        Get institutional activity analysis for a symbol
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict[str, Any]: Institutional activity analysis
        """
        try:
            # Try to get existing analysis
            analysis = self.institutional_activity_agent.get_institutional_activity(symbol)
            
            if analysis:
                # Check if analysis is recent (within 24 hours)
                analysis_date = datetime.fromisoformat(analysis.get("date", ""))
                if datetime.now() - analysis_date < timedelta(hours=24):
                    return {"symbol": symbol, "analysis": analysis}
            
            # Trigger new analysis if none exists or is outdated
            self._publish_user_request("institutional_activity", "analyze_institutional_for_symbol", {"symbol": symbol})
            
            # Return message that analysis is being generated
            return {
                "symbol": symbol,
                "status": "generating",
                "message": "Institutional activity analysis is being generated. Please check back in a few moments."
            }
        except Exception as e:
            logger.error(f"Error getting institutional activity for {symbol}: {e}")
            return {"error": str(e)}
    
    def get_risk_assessment(self, symbol: str = None) -> Dict[str, Any]:
        """
        Get risk assessment for a symbol or portfolio
        
        Args:
            symbol (str, optional): Stock symbol or None for portfolio assessment
            
        Returns:
            Dict[str, Any]: Risk assessment
        """
        try:
            # Try to get existing analysis
            analysis = self.risk_assessment_agent.get_risk_assessment(symbol)
            
            if analysis:
                # Check if analysis is recent (within 24 hours)
                analysis_date = datetime.fromisoformat(analysis.get("date", ""))
                if datetime.now() - analysis_date < timedelta(hours=24):
                    return {"symbol": symbol or "PORTFOLIO", "analysis": analysis}
            
            # Trigger new analysis if none exists or is outdated
            if symbol:
                self._publish_user_request("risk_assessment", "analyze_risk_for_symbol", {"symbol": symbol})
            else:
                self._publish_user_request("risk_assessment", "analyze_portfolio_risk", {"symbols": Config.DEFAULT_STOCK_SYMBOLS})
            
            # Return message that analysis is being generated
            return {
                "symbol": symbol or "PORTFOLIO",
                "status": "generating",
                "message": "Risk assessment is being generated. Please check back in a few moments."
            }
        except Exception as e:
            logger.error(f"Error getting risk assessment for {symbol or 'PORTFOLIO'}: {e}")
            return {"error": str(e)}
    
    def get_combined_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Get combined analysis for a symbol (market position, news sentiment, institutional activity, risk)
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict[str, Any]: Combined analysis
        """
        try:
            # Get stock data
            stock_data = self.get_stock_data(symbol)
            
            # Get company info
            company_info = self.get_company_info(symbol)
            
            # Get market position
            market_position = self.market_position_agent.get_market_position(symbol)
            
            # Get news sentiment
            news_sentiment = self.news_sentiment_agent.get_news_sentiment(symbol)
            
            # Get institutional activity
            institutional_activity = self.institutional_activity_agent.get_institutional_activity(symbol)
            
            # Get risk assessment
            risk_assessment = self.risk_assessment_agent.get_risk_assessment(symbol)
            
            # Combine all analyses
            combined_analysis = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "stock_data": stock_data.get("data", [])[-1] if "data" in stock_data else None,
                "company_info": company_info.get("company_info", {}),
                "market_position": market_position,
                "news_sentiment": news_sentiment,
                "institutional_activity": institutional_activity,
                "risk_assessment": risk_assessment
            }
            
            # Generate summary insights
            insights = self._generate_insights(combined_analysis)
            combined_analysis["insights"] = insights
            
            return combined_analysis
        except Exception as e:
            logger.error(f"Error getting combined analysis for {symbol}: {e}")
            return {"error": str(e)}
    
    def _generate_insights(self, combined_analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate insights from combined analysis
        
        Args:
            combined_analysis (Dict[str, Any]): Combined analysis data
            
        Returns:
            List[Dict[str, str]]: List of insights
        """
        insights = []
        
        # Extract key components
        symbol = combined_analysis.get("symbol", "")
        market_position = combined_analysis.get("market_position", {})
        news_sentiment = combined_analysis.get("news_sentiment", {})
        institutional_activity = combined_analysis.get("institutional_activity", {})
        risk_assessment = combined_analysis.get("risk_assessment", {})
        
        # Market position insights
        if market_position:
            position = market_position.get("overall_position", "")
            
            if position == "bullish":
                insights.append({
                    "type": "positive",
                    "category": "market_position",
                    "text": f"Technical indicators for {symbol} show a bullish trend"
                })
            elif position == "bearish":
                insights.append({
                    "type": "negative",
                    "category": "market_position",
                    "text": f"Technical indicators for {symbol} show a bearish trend"
                })
            elif position == "potential_bullish_reversal":
                insights.append({
                    "type": "opportunity",
                    "category": "market_position",
                    "text": f"Technical indicators suggest a potential bullish reversal for {symbol}"
                })
        
        # News sentiment insights
        if news_sentiment:
            sentiment = news_sentiment.get("overall_sentiment", "")
            
            if sentiment == "positive":
                insights.append({
                    "type": "positive",
                    "category": "news_sentiment",
                    "text": f"Recent news sentiment for {symbol} is positive"
                })
            elif sentiment == "negative":
                insights.append({
                    "type": "negative",
                    "category": "news_sentiment",
                    "text": f"Recent news sentiment for {symbol} is negative"
                })
        
        # Institutional activity insights
        if institutional_activity:
            inst_ownership = institutional_activity.get("institutional_ownership_percentage", 0)
            
            if inst_ownership > 70:
                insights.append({
                    "type": "positive",
                    "category": "institutional_activity",
                    "text": f"Strong institutional ownership ({inst_ownership:.1f}%) for {symbol}"
                })
        
        # Risk assessment insights
        if risk_assessment:
            risk_level = risk_assessment.get("risk_level", "")
            risk_factors = risk_assessment.get("risk_factors", [])
            
            if risk_level == "high":
                insights.append({
                    "type": "warning",
                    "category": "risk_assessment",
                    "text": f"High risk level detected for {symbol}"
                })
            
            if "high_volatility" in risk_factors:
                insights.append({
                    "type": "warning",
                    "category": "risk_assessment",
                    "text": f"{symbol} shows high volatility"
                })
        
        # Combined insights
        if market_position and news_sentiment:
            market_pos = market_position.get("overall_position", "")
            news_sent = news_sentiment.get("overall_sentiment", "")
            
            if market_pos == "bullish" and news_sent == "positive":
                insights.append({
                    "type": "opportunity",
                    "category": "combined",
                    "text": f"Strong buy signal: Bullish technical indicators with positive news sentiment for {symbol}"
                })
            elif market_pos == "bearish" and news_sent == "negative":
                insights.append({
                    "type": "warning",
                    "category": "combined",
                    "text": f"Strong sell signal: Bearish technical indicators with negative news sentiment for {symbol}"
                })
        
        return insights
    
    def _publish_user_request(self, target: str, request_type: str, data: Dict[str, Any] = None):
        """
        Publish a user request event to the event bus
        
        Args:
            target (str): Target agent or service
            request_type (str): Type of request
            data (Dict[str, Any], optional): Additional request data
        """
        event_data = {
            "target": target,
            "request_type": request_type,
            "timestamp": datetime.now().isoformat()
        }
        
        if data:
            event_data.update(data)
        
        event = Event(
            event_type=EventType.USER_REQUEST,
            data=event_data,
            source="GraphService"
        )
        
        self.event_bus.publish(event)
        logger.info(f"Published user request: {event}")
    
    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all agents
        
        Returns:
            Dict[str, Dict[str, Any]]: Status of all agents
        """
        return {
            "market_position_agent": self.market_position_agent.get_status(),
            "news_sentiment_agent": self.news_sentiment_agent.get_status(),
            "institutional_activity_agent": self.institutional_activity_agent.get_status(),
            "risk_assessment_agent": self.risk_assessment_agent.get_status()
        }
