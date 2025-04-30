from enum import Enum
from typing import Dict, Any, List
from pydantic import BaseModel
from datetime import datetime

class AgentType(Enum):
    MARKET_POSITION = "market_position"
    NEWS_SENTIMENT = "news_sentiment"
    INSTITUTIONAL_ACTIVITY = "institutional_activity"
    RISK_ASSESSMENT = "risk_assessment"

class MetricType(Enum):
    FINANCIAL_RATIO = "financial_ratio"
    SENTIMENT_SCORE = "sentiment_score"
    OWNERSHIP_METRIC = "ownership_metric"
    RISK_SCORE = "risk_score"
    COMPOSITE_SCORE = "composite_score"
    PE_RATIO = "pe_ratio"
    PB_RATIO = "pb_ratio"
    RELATIVE_STRENGTH = "relative_strength"
    OWNERSHIP_CONCENTRATION = "ownership_concentration"
    NET_OWNERSHIP_CHANGE = "net_ownership_change"
    CONFIDENCE_SCORE = "confidence_score"
    COMPOSITE_RISK_SCORE = "composite_risk_score"

class AnalysisTimeframe(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

class AgentMessage(BaseModel):
    agent_type: str
    data: Dict[str, Any]
    timestamp: str

class MarketPositionMetrics(BaseModel):
    pe_ratio: float
    pb_ratio: float
    relative_strength: float

class NewsSentiment(BaseModel):
    sentiment_score: float
    headline: str = ""
    source: str

class InstitutionalOwnershipMetrics(BaseModel):
    ownership_concentration: float
    net_ownership_change: float
    confidence_score: float

class RiskAssessmentMetrics(BaseModel):
    composite_risk_score: float
    risk_factors: Dict[str, float]

# Mapping of actual available endpoints for each agent type
# Updated based on API testing results
AGENT_ENDPOINTS: Dict[str, List[str]] = {
    AgentType.MARKET_POSITION.value: [
        "get_stock_info",  # This endpoint works (company/facts)
        "calculate_financial_ratios"  # Derived from company facts data
    ],
    AgentType.NEWS_SENTIMENT.value: [
        "get_news"  # We can use Alpha Vantage for this
    ],
    AgentType.INSTITUTIONAL_ACTIVITY.value: [
        "calculate_institutional_metrics"  # Derived from company facts data
    ],
    AgentType.RISK_ASSESSMENT.value: []  # Aggregates data from other agents
}

# Define data sources with fallback paths
DATA_SOURCES = {
    "company_facts": {
        "primary": "financial_datasets_api",
        "endpoint": "/company/facts",
        "fallback": "mock_generator"
    },
    "market_position": {
        "primary": "derived_calculation", 
        "source": "company_facts",
        "fallback": "mock_generator"
    },
    "financial_ratios": {
        "primary": "derived_calculation",
        "source": "company_facts",
        "fallback": "mock_generator"
    },
    "news": {
        "primary": "alpha_vantage_api",
        "endpoint": "/query?function=NEWS_SENTIMENT",
        "fallback": "tavily_api"
    }
}

# Default weights for risk assessment scoring
RISK_FACTOR_WEIGHTS: Dict[str, float] = {
    "market_position": 0.3,
    "sentiment": 0.2,
    "institutional_activity": 0.3,
    "market_volatility": 0.2
}

# Confidence score thresholds
CONFIDENCE_THRESHOLDS = {
    "HIGH": 0.8,
    "MEDIUM": 0.6,
    "LOW": 0.4
}

# News sentiment scoring parameters
SENTIMENT_PARAMS = {
    "recent_weight": 0.6,
    "historical_weight": 0.4,
    "keyword_impact_threshold": 0.3,
    "time_decay_factor": 0.9
}

# Institutional activity analysis parameters
INSTITUTIONAL_PARAMS = {
    "ownership_change_threshold": 0.05,
    "significant_transaction_threshold": 1000000,
    "insider_weight": 0.4,
    "institutional_weight": 0.6
}

# Market position analysis parameters
MARKET_POSITION_PARAMS = {
    "pe_ratio_weight": 0.3,
    "pb_ratio_weight": 0.2,
    "revenue_growth_weight": 0.25,
    "profit_margin_weight": 0.25
}

# Standard event topics for agent communication
EVENT_TOPICS = {
    "market_metrics": "market_position.metrics",
    "news_sentiment": "news_sentiment.update",
    "institutional_metrics": "institutional.metrics",
    "risk_assessment": "risk.assessment"
}

# API configuration keys
REQUIRED_API_CONFIGS = {
    "alpha_vantage_key": "Alpha Vantage API key",
    "financial_api_key": "Financial Data API key",
    "openai_api_key": "OpenAI API key"
}

# GPT Analysis Prompts
GPT_PROMPTS = {
    "market_analysis": """Analyze the following market metrics and provide insights:
- PE Ratio: {pe_ratio}
- PB Ratio: {pb_ratio}
- Relative Strength: {relative_strength}
Consider industry standards and current market conditions.""",

    "news_sentiment": """Analyze the following news headlines and provide sentiment analysis:
Headlines: {headlines}
Consider market impact, credibility, and temporal relevance.""",

    "institutional_analysis": """Analyze institutional trading patterns:
- Ownership Concentration: {concentration}
- Net Changes: {changes}
- Recent Transactions: {transactions}
Consider insider trading patterns and institutional behavior trends.""",

    "risk_assessment": """Provide a comprehensive risk assessment based on:
Market Position: {market_data}
News Sentiment: {sentiment_data}
Institutional Activity: {institutional_data}
Consider all factors and their interrelations."""
}

# GPT Model Configuration
GPT_CONFIG = {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 500
}
