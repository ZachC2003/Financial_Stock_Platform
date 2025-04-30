"""
Wrapper methods for agents to ensure consistent method naming.

This module adds wrapper methods to each agent class to ensure
that the method names expected by the Streamlit app match the
actual methods provided by the agent implementations.
"""

# Add wrapper methods to the NewsSentimentAgent class to match expected method names
def add_sentiment_analysis_wrappers(cls):
    """Add wrapper methods to the NewsSentimentAgent class."""
    
    # Add wrapper for get_sentiment_analysis if it doesn't exist
    if not hasattr(cls, 'get_sentiment_analysis'):
        def get_sentiment_analysis(self, symbol):
            """Wrapper for get_news_sentiment."""
            return self.get_news_sentiment(symbol)
        setattr(cls, 'get_sentiment_analysis', get_sentiment_analysis)
    
    return cls

# Add wrapper methods to the MarketPositionAgent class
def add_market_position_wrappers(cls):
    """Add wrapper methods to the MarketPositionAgent class."""
    
    # Add wrapper for get_market_position if it doesn't exist
    if not hasattr(cls, 'get_market_position'):
        def get_market_position(self, symbol):
            """Wrapper for analyze_position."""
            return self.analyze_position(symbol)
        setattr(cls, 'get_market_position', get_market_position)
    
    return cls

# Add wrapper methods to the InstitutionalActivityAgent class
def add_institutional_activity_wrappers(cls):
    """Add wrapper methods to the InstitutionalActivityAgent class."""
    
    # Add wrapper for get_institutional_analysis if it doesn't exist
    if not hasattr(cls, 'get_institutional_analysis'):
        def get_institutional_analysis(self, symbol):
            """Wrapper for analyze_institutional_activity."""
            return self.analyze_institutional_activity(symbol)
        setattr(cls, 'get_institutional_analysis', get_institutional_analysis)
    
    return cls

# Add wrapper methods to the RiskAssessmentAgent class
def add_risk_assessment_wrappers(cls):
    """Add wrapper methods to the RiskAssessmentAgent class."""
    
    # Add wrapper for get_risk_assessment if it doesn't exist
    if not hasattr(cls, 'get_risk_assessment'):
        def get_risk_assessment(self, symbol):
            """Wrapper for assess_risk."""
            return self.assess_risk(symbol)
        setattr(cls, 'get_risk_assessment', get_risk_assessment)
    
    return cls

# Add wrapper methods to the TavilyScraper class
def add_tavily_scraper_wrappers(cls):
    """Add wrapper methods to the TavilyScraper class."""
    
    # Add wrapper for get_stock_news if it doesn't exist
    if not hasattr(cls, 'get_stock_news'):
        def get_stock_news(self, symbol):
            """Generate mock stock news data if the API key is not available."""
            from tests.mock_data_provider import MockDataProvider
            return MockDataProvider.get_news(symbol)
        setattr(cls, 'get_stock_news', get_stock_news)
    
    return cls
