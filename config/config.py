import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys and Configuration
class Config:
    # Supabase Configuration
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    # Alpha Vantage API
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    
    # Financial Datasets API
    FINANCIAL_API_KEY = os.getenv("FINANCIAL_API_KEY")
    
    # Tavily API for web scraping
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    # OpenAI API for NLP and analysis
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # WRDS API (if applicable)
    WRDS_USERNAME = os.getenv("WRDS_USERNAME")
    WRDS_PASSWORD = os.getenv("WRDS_PASSWORD")
    
    # Default settings
    DEFAULT_STOCK_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    DEFAULT_TIME_PERIOD = "1y"  # 1 year
    
    # Database settings
    DB_STOCK_DATA_TABLE = "stock_data"
    DB_NEWS_DATA_TABLE = "news_data"
    DB_ANALYSIS_TABLE = "analysis_results"
    
    # Agent settings
    AGENT_POLLING_INTERVAL = 3600  # in seconds (1 hour)
    
    @staticmethod
    def validate_config():
        """Validate that all required configuration variables are set"""
        required_vars = [
            "SUPABASE_URL", 
            "SUPABASE_KEY", 
            "ALPHA_VANTAGE_API_KEY",
            "FINANCIAL_API_KEY",
            "TAVILY_API_KEY",
            "OPENAI_API_KEY"
        ]
        
        missing_vars = []
        for var in required_vars:
            if getattr(Config, var) is None:
                missing_vars.append(var)
        
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True
