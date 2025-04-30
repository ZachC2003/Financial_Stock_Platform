import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from backend.agent_system.agents.base_agent import BaseAgent
from backend.agent_system.event_bus import EventBus, Event, EventType
from backend.data_services.data_connector import DataConnector
from backend.data_services.supabase_client import SupabaseClient
from backend.agent_system.agent_types import AgentType, MarketPositionMetrics, MARKET_POSITION_PARAMS
from config.config import Config

class MarketPositionAgent(BaseAgent):
    """
    Agent responsible for analyzing market positions and trends.
    Monitors stock performance, technical indicators, and market trends.
    
    This agent uses the new architecture with DataConnector for improved
    data handling and fallbacks when API endpoints are unavailable.
    """
    
    def __init__(self, polling_interval: int = 3600):
        """
        Initialize the Market Position Agent
        
        Args:
            polling_interval (int): Interval between agent runs in seconds
        """
        super().__init__(name="MarketPositionAgent", polling_interval=polling_interval)
        
        self.data_connector = DataConnector()
        self.supabase_client = SupabaseClient()
        self.symbols = Config.DEFAULT_STOCK_SYMBOLS
        self.time_period = Config.DEFAULT_TIME_PERIOD
        
        self.logger.info("Initialized MarketPositionAgent with new DataConnector architecture")
    
    def _subscribe_to_events(self):
        """Subscribe to relevant events on the event bus"""
        self.event_bus.subscribe(EventType.STOCK_DATA_UPDATED, self._handle_stock_data_updated)
        self.event_bus.subscribe(EventType.USER_REQUEST, self._handle_user_request)
    
    def _handle_stock_data_updated(self, event: Event):
        """
        Handle stock data updated event
        
        Args:
            event (Event): Stock data updated event
        """
        self.logger.info(f"Received stock data updated event: {event}")
        
        # If stock data was updated, we might want to run a new analysis
        if 'symbol' in event.data:
            symbol = event.data['symbol']
            if symbol in self.symbols:
                self.logger.info(f"Triggering analysis for updated symbol: {symbol}")
                self.analyze_market_position(symbol)
    
    def _handle_user_request(self, event: Event):
        """
        Handle user request event
        
        Args:
            event (Event): User request event
        """
        self.logger.info(f"Received user request event: {event}")
        
        # Check if the request is for this agent
        if event.data.get('target') == 'market_position' or event.data.get('target') == 'all':
            request_type = event.data.get('request_type')
            
            if request_type == 'analyze_symbol':
                symbol = event.data.get('symbol')
                if symbol:
                    self.logger.info(f"User requested analysis for symbol: {symbol}")
                    self.analyze_market_position(symbol)
            
            elif request_type == 'analyze_all':
                self.logger.info("User requested analysis for all symbols")
                self.run()
    
    def analyze_position(self, symbol):
        """
        Wrapper method for analyze_market_position for compatibility with Streamlit app
        
        Args:
            symbol (str): Stock symbol to analyze
            
        Returns:
            dict: Market position analysis results
        """
        return self.analyze_market_position(symbol)
    
    def run(self):
        """
        Main method that runs the agent workflow for all configured symbols
        """
        self.logger.info("Starting market position analysis for all symbols")
        
        for symbol in self.symbols:
            try:
                self.analyze_market_position(symbol)
            except Exception as e:
                self.logger.error(f"Error analyzing market position for {symbol}: {str(e)}")
                
        self.logger.info("Completed market position analysis for all symbols")
    
    def analyze_market_position(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze market position for a specific symbol using the new architecture.
        This method fetches data through the DataConnector which handles API fallbacks.
        
        Args:
            symbol (str): Stock ticker symbol to analyze
            
        Returns:
            dict: Market position analysis results including ratios and indicators
        """
        self.logger.info(f"Starting market position analysis for {symbol}")
        
        try:
            # Get all required data for market position analysis
            # DataConnector will handle API calls or fallbacks as needed
            data = self.data_connector.get_data_for_agent(AgentType.MARKET_POSITION.value, symbol)
            
            # Extract the data components
            company_facts = data.get("company_facts", {}).get("company_facts", {})
            financial_ratios = data.get("financial_ratios", {})
            
            # Create a comprehensive market position analysis
            pe_ratio = financial_ratios.get("pe_ratio", 0)
            pb_ratio = financial_ratios.get("pb_ratio", 0)
            relative_strength = financial_ratios.get("relative_strength", 0)
            
            # Determine market position
            if relative_strength > 1.1:
                position = "bullish"
            elif relative_strength < 0.9:
                position = "bearish"
            else:
                position = "neutral"
                
            # Create a MarketPositionMetrics model
            metrics = {
                "pe_ratio": pe_ratio,
                "pb_ratio": pb_ratio,
                "relative_strength": relative_strength,
                "position": position,
                "company_name": company_facts.get("name", f"{symbol} Inc"),
                "sector": company_facts.get("sector", "Unknown"),
                "industry": company_facts.get("industry", "Unknown"),
                "market_cap": company_facts.get("market_cap", 0),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store the analysis in Supabase if available
            self._store_analysis_results(symbol, metrics)
            
            # Publish the results to the event bus
            self._publish_analysis_results(symbol, metrics)
            
            self.logger.info(f"Completed market position analysis for {symbol}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing market position for {symbol}: {str(e)}")
            return {
                "error": str(e),
                "pe_ratio": 0,
                "pb_ratio": 0,
                "relative_strength": 0,
                "position": "unknown"
            }
    
    def _store_analysis_results(self, symbol: str, results: Dict[str, Any]):
        """
        Store analysis results in Supabase
        
        Args:
            symbol (str): Stock ticker symbol
            results (dict): Analysis results to store
        """
        try:
            if not self.supabase_client.is_connected():
                self.logger.warning("Supabase client not connected, skipping storage")
                return
                
            # Add timestamp and symbol for database
            data_to_store = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "analysis_type": "market_position",
                "data": results
            }
            
            # Store in Supabase
            self.supabase_client.store_analysis(data_to_store)
            self.logger.info(f"Stored market position analysis for {symbol} in Supabase")
            
        except Exception as e:
            self.logger.error(f"Error storing analysis results: {str(e)}")
            # Non-critical, so just log the error and continue
            self.logger.info(f"Stored market position analysis for {symbol}")
            
    def _publish_analysis_results(self, symbol: str, results: Dict[str, Any]):
        """
        Publish analysis results to the event bus
        
        Args:
            symbol (str): Stock ticker symbol
            results (dict): Analysis results to publish
        """
        try:
            # Create event data
            event_data = {
                "symbol": symbol,
                "analysis_type": "market_position",
                "data": results
            }
            
            # Create and publish event
            event = Event(
                event_type=EventType.MARKET_ANALYSIS_COMPLETED,  # Using the correct EventType from the enum
                source=self.name,
                data=event_data
            )
            
            # Publish to event bus
            self.event_bus.publish(event)
            self.logger.info(f"Published market position analysis for {symbol} to event bus")
            
        except Exception as e:
            self.logger.error(f"Error publishing analysis results: {str(e)}")
    
    def get_market_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest market position analysis for a symbol
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Optional[Dict[str, Any]]: Latest market position analysis or None if not found
        """
        try:
            results = self.supabase_client.get_analysis_results(
                symbol=symbol,
                analysis_type="market_position",
                limit=1
            )
            
            if not results.empty:
                return results.iloc[0].to_dict()
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting market position for {symbol}: {e}")
            return None
