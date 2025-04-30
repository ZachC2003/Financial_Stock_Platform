"""
Stock Server Client Module

This module provides a client wrapper for the stock server API.
It handles the communication between the application and the stock server.
"""

import os
import logging
import requests
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class StockClient:
    """Client class for stock server access.
    This wrapper provides client-side access to the FastMCP stock server functions.
    """
    def __init__(self, host=None):
        # Use environment variable if available, otherwise use default
        self.host = host or os.getenv('STOCK_SERVER_URL', 'http://localhost:8000')
        self.endpoint = f"{self.host}/api/stock"
        logger.info(f"Initializing StockClient with endpoint: {self.endpoint}")
    
    def get_stock_info(self, ticker):
        """Get stock information for the given ticker."""
        try:
            response = requests.post(f"{self.endpoint}/get_stock_info", json={"ticker": ticker})
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching stock info: {e}")
            return {"success": False, "error": str(e)}
    
    def get_financial_snapshot(self, ticker):
        """Get financial metrics snapshot for a ticker."""
        try:
            response = requests.post(f"{self.endpoint}/get_financial_snapshot", json={"ticker": ticker})
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching financial snapshot: {e}")
            return {"success": False, "error": str(e)}
    
    def get_insider_trades(self, ticker, limit=100):
        """Get insider trading information for a ticker."""
        try:
            response = requests.post(f"{self.endpoint}/get_insider_trades", 
                                   json={"ticker": ticker, "limit": limit})
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching insider trades: {e}")
            return {"success": False, "error": str(e)}
    
    def get_news(self, ticker, days_back=7, limit=10):
        """Get news articles for a ticker."""
        try:
            response = requests.post(f"{self.endpoint}/get_news", 
                                   json={"ticker": ticker, "days_back": days_back, "limit": limit})
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return {"success": False, "error": str(e)}
    
    def get_institutional_ownership(self, ticker, limit=100):
        """Get institutional ownership information for a ticker."""
        try:
            response = requests.post(f"{self.endpoint}/get_institutional_ownership", 
                                   json={"ticker": ticker, "limit": limit})
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching institutional ownership: {e}")
            return {"success": False, "error": str(e)}
    
    # Alias for get_institutional_ownership for compatibility
    def get_institutional_holdings(self, ticker, limit=100):
        """Alias for get_institutional_ownership for compatibility."""
        return self.get_institutional_ownership(ticker, limit)
    
    def get_historical_data(self, ticker, period="1y", interval="1d"):
        """
        Get historical price data for a ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): Time period to retrieve data for (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max')
            interval (str): Data interval (e.g., '1d', '1wk', '1mo')
            
        Returns:
            dict: Dictionary containing historical price data
        """
        try:
            response = requests.post(f"{self.endpoint}/get_historical_data", 
                                   json={"ticker": ticker, "period": period, "interval": interval})
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return {"success": False, "error": str(e)}
    
    def calculate_technical_indicators(self, ticker, period="1y", interval="1d"):
        """
        Calculate technical indicators for a ticker based on historical data.
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): Time period to analyze
            interval (str): Data interval
            
        Returns:
            dict: Dictionary containing technical indicators
        """
        # First get historical data
        hist_data = self.get_historical_data(ticker, period, interval)
        
        # If there was an error or no data, return early
        if not hist_data.get('success', False) or not hist_data.get('data', {}).get('prices', []):
            return {"success": False, "error": "No historical data available for technical indicators"}
        
        # Extract price data
        prices = hist_data['data']['prices']
        
        try:
            # Calculate indicators
            result = {
                "success": True,
                "data": {
                    "ticker": ticker,
                    "period": period,
                    "interval": interval,
                    "indicators": {}
                }
            }
            
            # Add calculated indicators here
            # You would typically call methods to calculate each indicator
            
            return result
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {"success": False, "error": str(e)}
    
    def get_major_holders(self, ticker):
        """
        Get major holders information for a ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Dictionary containing major holders information
        """
        try:
            response = requests.post(f"{self.endpoint}/get_major_holders", json={"ticker": ticker})
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching major holders: {e}")
            return {"success": False, "error": str(e)}
    
    def get_earnings_releases(self, ticker, days_back=365, limit=10):
        """
        Get earnings press releases for a specific company ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            days_back (int): Number of days to look back from today (default: 365)
            limit (int): Number of earnings releases to return (default: 10)
            
        Returns:
            dict: Dictionary containing earnings releases information
        """
        try:
            response = requests.post(f"{self.endpoint}/get_earnings_releases", 
                                   json={"ticker": ticker, "days_back": days_back, "limit": limit})
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching earnings releases: {e}")
            return {"success": False, "error": str(e)}
    
    def get_market_position(self, ticker):
        """Get market position analysis for a ticker."""
        try:
            response = requests.post(f"{self.endpoint}/get_market_position", json={"ticker": ticker})
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching market position: {e}")
            return {"success": False, "error": str(e)}
    
    def get_sentiment_analysis(self, ticker):
        """Get sentiment analysis for a ticker's news and social media presence."""
        try:
            response = requests.post(f"{self.endpoint}/get_sentiment_analysis", json={"ticker": ticker})
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching sentiment analysis: {e}")
            return {"success": False, "error": str(e)}
    
    def get_institutional_analysis(self, ticker):
        """Get institutional activity analysis for a ticker."""
        try:
            response = requests.post(f"{self.endpoint}/get_institutional_analysis", json={"ticker": ticker})
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching institutional analysis: {e}")
            return {"success": False, "error": str(e)}
            
    def get_historical_data(self, ticker, period="1y", interval="1d"):
        """Get historical price data for a ticker."""
        try:
            response = requests.post(f"{self.endpoint}/get_historical_data", 
                                   json={"ticker": ticker, "period": period, "interval": interval})
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return {"success": False, "error": str(e)}
    
    def get_company_info(self, ticker):
        """Get company information for a ticker (wrapper for get_stock_info)."""
        return self.get_stock_info(ticker)
    
    def get_risk_assessment(self, ticker):
        """Get risk assessment metrics for a ticker."""
        try:
            response = requests.post(f"{self.endpoint}/get_risk_assessment", json={"ticker": ticker})
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching risk assessment: {e}")
            return {"success": False, "error": str(e)}
    
    def get_earnings_data(self, ticker):
        """Get earnings data for a ticker (alias for get_earnings_releases)."""
        return self.get_earnings_releases(ticker)
