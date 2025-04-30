"""
Mock data provider for testing the Streamlit app without consuming API credits.
This module provides sample data that mimics the structure of real API responses.
"""

import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

class MockDataProvider:
    """Provides mock data for testing without making actual API calls."""
    
    @staticmethod
    def get_financial_snapshot(symbol: str) -> Dict[str, Any]:
        """Mock financial snapshot data."""
        return {
            'success': True,
            'data': {
                'symbol': symbol,
                'name': f"{symbol} Corporation",
                'exchange': 'NASDAQ',
                'market_cap': random.uniform(10000000000, 2000000000000),
                'sector': random.choice(['Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical']),
                'industry': 'Software—Application',
                'price': random.uniform(50, 500),
                'price_to_earnings_ratio': random.uniform(10, 40),
                'price_to_book_ratio': random.uniform(2, 15),
                'price_to_sales_ratio': random.uniform(1, 20),
                'enterprise_value_to_ebitda_ratio': random.uniform(8, 25),
                'gross_margin': random.uniform(0.3, 0.8),
                'operating_margin': random.uniform(0.15, 0.4),
                'net_margin': random.uniform(0.05, 0.3),
                'return_on_equity': random.uniform(0.05, 0.4),
                'return_on_assets': random.uniform(0.03, 0.2),
                'current_ratio': random.uniform(1, 3),
                'quick_ratio': random.uniform(0.8, 2.5),
                'debt_to_equity': random.uniform(0.1, 2),
                'interest_coverage': random.uniform(5, 30),
                'revenue_growth': random.uniform(-0.05, 0.3),
                'earnings_growth': random.uniform(-0.1, 0.4),
                'earnings_per_share_growth': random.uniform(-0.1, 0.35),
                'dividend_yield': random.uniform(0, 0.04),
                'payout_ratio': random.uniform(0, 0.7),
                'beta': random.uniform(0.5, 2),
                'fifty_two_week_high': random.uniform(300, 700),
                'fifty_two_week_low': random.uniform(100, 299),
                'moving_average_50day': random.uniform(200, 500),
                'moving_average_200day': random.uniform(200, 500),
                'shares_outstanding': random.uniform(1000000000, 10000000000),
                'shares_float': random.uniform(900000000, 9000000000),
                'percent_insiders': random.uniform(0.01, 0.2),
                'percent_institutions': random.uniform(0.4, 0.9),
                'short_ratio': random.uniform(1, 10),
                'book_value_growth': random.uniform(-0.05, 0.2),
                'operating_cash_flow_growth': random.uniform(-0.1, 0.3),
                'free_cash_flow_growth': random.uniform(-0.15, 0.25),
                'last_updated': datetime.now().isoformat()
            }
        }
    
    @staticmethod
    def get_insider_trades(symbol: str) -> Dict[str, Any]:
        """Mock insider trades data."""
        insiders = [
            {"name": "John Smith", "title": "CEO", "is_board_director": True},
            {"name": "Jane Doe", "title": "CFO", "is_board_director": True},
            {"name": "Robert Johnson", "title": "CTO", "is_board_director": True},
            {"name": "Emily Williams", "title": "COO", "is_board_director": True},
            {"name": "Michael Brown", "title": "Director", "is_board_director": True},
            {"name": "Lisa Davis", "title": "VP of Sales", "is_board_director": False},
            {"name": "David Wilson", "title": "VP of Marketing", "is_board_director": False},
            {"name": "Sarah Miller", "title": "VP of Engineering", "is_board_director": False}
        ]
        
        trades = []
        for _ in range(20):
            insider = random.choice(insiders)
            transaction_shares = random.choice([-10000, -5000, -1000, 1000, 5000, 10000])
            transaction_value = abs(transaction_shares) * random.uniform(50, 200)
            transaction_date = (datetime.now() - timedelta(days=random.randint(1, 120))).strftime("%Y-%m-%d")
            
            trades.append({
                **insider,
                "symbol": symbol,
                "transaction_date": transaction_date,
                "transaction_type": "BUY" if transaction_shares > 0 else "SELL",
                "transaction_shares": transaction_shares,
                "transaction_value": transaction_value,
                "shares_owned_after": random.randint(10000, 1000000)
            })
        
        return {
            'success': True,
            'data': trades
        }
    
    @staticmethod
    def get_institutional_ownership(symbol: str) -> Dict[str, Any]:
        """Mock institutional ownership data."""
        institutions = [
            "BlackRock Inc.",
            "Vanguard Group Inc.",
            "State Street Corporation",
            "Fidelity Management & Research",
            "Capital Research & Management",
            "Wellington Management",
            "JPMorgan Chase & Co.",
            "Bank of America Corporation",
            "Morgan Stanley",
            "Goldman Sachs Group Inc.",
            "T. Rowe Price Associates",
            "Geode Capital Management"
        ]
        
        ownership_data = []
        total_shares = random.uniform(900000000, 1000000000)
        shares_allocated = 0
        
        for institution in institutions:
            if shares_allocated >= total_shares * 0.8:
                break
                
            shares = random.uniform(total_shares * 0.01, total_shares * 0.15)
            shares_allocated += shares
            
            ownership_data.append({
                "institution": institution,
                "shares": int(shares),
                "market_value": int(shares * random.uniform(50, 500)),
                "percentage": shares / total_shares,
                "change": random.uniform(-0.2, 0.3),
                "filing_date": (datetime.now() - timedelta(days=random.randint(30, 90))).strftime("%Y-%m-%d")
            })
        
        return {
            'success': True,
            'data': ownership_data
        }
    
    @staticmethod
    def get_news(symbol: str) -> Dict[str, Any]:
        """Mock news data."""
        news_headlines = [
            f"{symbol} Reports Record Quarterly Earnings",
            f"{symbol} Announces New Product Line",
            f"{symbol} Expands into International Markets",
            f"{symbol} CEO Discusses Future Growth Strategy",
            f"Analysts Upgrade {symbol} Stock Rating",
            f"{symbol} Completes Acquisition of Tech Startup",
            f"Institutional Investors Increase Stakes in {symbol}",
            f"{symbol} Beats Market Expectations in Q2",
            f"New Partnership Announced Between {symbol} and Industry Leader",
            f"{symbol} Faces Regulatory Scrutiny Over Business Practices"
        ]
        
        news_data = []
        for i in range(min(10, len(news_headlines))):
            days_ago = random.randint(1, 30)
            news_date = (datetime.now() - timedelta(days=days_ago)).isoformat()
            sentiment = random.choice(["positive", "neutral", "negative"])
            source = random.choice(["Bloomberg", "Reuters", "CNBC", "Wall Street Journal", "Financial Times"])
            
            news_data.append({
                "title": news_headlines[i],
                "date": news_date,
                "source": source,
                "url": f"https://example.com/news/{symbol.lower()}/{i}",
                "snippet": f"This is a sample news snippet about {symbol}. The company has been making headlines recently...",
                "author": f"Financial Reporter {i+1}",
                "sentiment": sentiment
            })
        
        return {
            'success': True,
            'data': news_data,
            'date_range': {
                'start_date': (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                'end_date': datetime.now().strftime("%Y-%m-%d")
            }
        }
    
    @staticmethod
    def get_news_sentiment(symbol: str):
        """Mock news sentiment data."""
        class MockSentiment:
            def __init__(self, symbol):
                self.symbol = symbol
                self.sentiment_score = random.uniform(-0.8, 0.8)
                self.confidence = random.uniform(0.6, 0.95)
                self.headline = f"Investors React to {symbol}'s Latest Announcement"
                self.sentiment_breakdown = {
                    "positive": random.uniform(0.2, 0.8),
                    "neutral": random.uniform(0.1, 0.4),
                    "negative": random.uniform(0.1, 0.5)
                }
                self.gpt_analysis = f"Recent news about {symbol} shows a trend toward {'positive' if self.sentiment_score > 0 else 'negative'} sentiment. The company's recent announcements on {'product innovations' if self.sentiment_score > 0 else 'financial challenges'} have driven this shift in market perception."
        
        return MockSentiment(symbol)
        
    @staticmethod
    def get_sentiment_analysis(symbol: str = None):
        """Wrapper for get_news_sentiment to match the API expected by the mock NewsSentimentAgent."""
        if not symbol:
            symbol = "AAPL"
        return MockDataProvider.get_news_sentiment(symbol)
    
    @staticmethod
    def analyze_position(symbol: str):
        """Mock market position analysis data."""
        class MockPositionData:
            def __init__(self, symbol):
                self.symbol = symbol
                self.pe_ratio = random.uniform(10, 40)
                self.pb_ratio = random.uniform(2, 15)
                self.relative_strength = random.uniform(-20, 30)
                self.sector_rank = random.randint(1, 20)
                self.market_cap_rank = random.randint(1, 100)
                self.gpt_analysis = f"{symbol} currently {'trades at a premium compared to' if self.pe_ratio > 25 else 'is valued favorably against'} industry peers. The company shows {'strong' if self.relative_strength > 0 else 'weak'} price momentum relative to the market. Based on fundamental metrics, this stock appears to be {'overvalued' if self.pe_ratio > 30 and self.pb_ratio > 10 else 'fairly valued' if self.pe_ratio > 20 else 'undervalued'}."
        
        return MockPositionData(symbol)
    
    @staticmethod
    def get_market_position(symbol: str = None):
        """Wrapper for analyze_position to match the API expected by the mock MarketPositionAgent."""
        # If no symbol provided, use a default
        if not symbol:
            symbol = "AAPL"
        return MockDataProvider.analyze_position(symbol)
    
    @staticmethod
    def analyze_institutional_activity(symbol: str):
        """Mock institutional activity analysis data."""
        class MockInstitutionalData:
            def __init__(self, symbol):
                self.symbol = symbol
                self.ownership_concentration = random.uniform(0.1, 0.9)
                self.net_ownership_change = random.uniform(-15, 20)
                self.confidence_score = random.uniform(0.1, 1.0)
                self.largest_holders = [
                    {"name": "BlackRock Inc.", "ownership": random.uniform(0.05, 0.15)},
                    {"name": "Vanguard Group Inc.", "ownership": random.uniform(0.04, 0.12)},
                    {"name": "State Street Corporation", "ownership": random.uniform(0.02, 0.08)}
                ]
                self.gpt_analysis = f"Institutional investors have {'increased' if self.net_ownership_change > 0 else 'decreased'} their positions in {symbol} by approximately {abs(self.net_ownership_change):.1f}% over the last quarter. This {'accumulation' if self.net_ownership_change > 0 else 'reduction'} suggests {'growing confidence' if self.net_ownership_change > 0 else 'some concerns'} about the company's future prospects. The {'high' if self.ownership_concentration > 0.5 else 'moderate' if self.ownership_concentration > 0.3 else 'low'} concentration of ownership among key institutions indicates {'potential vulnerability to large sell-offs' if self.ownership_concentration > 0.7 else 'a balanced ownership structure'}."
        
        return MockInstitutionalData(symbol)
        
    @staticmethod
    def get_institutional_analysis(symbol: str = None):
        """Wrapper for analyze_institutional_activity to match the API expected by the mock InstitutionalActivityAgent."""
        if not symbol:
            symbol = "AAPL"
        return MockDataProvider.analyze_institutional_activity(symbol)
    
    @staticmethod
    def assess_risk(symbol: str):
        """Mock risk assessment data with enhanced Tavily integration."""
        class MockRiskData:
            def __init__(self, symbol):
                self.symbol = symbol
                self.composite_risk_score = random.uniform(0.1, 0.9)
                
                # Original risk factors
                self.risk_factors = {
                    "Market Risk": random.uniform(0.1, 0.9),
                    "Financial Leverage Risk": random.uniform(0.1, 0.9),
                    "Operational Risk": random.uniform(0.1, 0.9),
                    "Valuation Risk": random.uniform(0.1, 0.9),
                    "Liquidity Risk": random.uniform(0.1, 0.9),
                    "News Sentiment Risk": random.uniform(0.1, 0.9),
                    "Keyword Risk": random.uniform(0.1, 0.9)
                }
                
                self.metrics = {
                    "beta": random.uniform(0.5, 2.0),
                    "volatility_30d": random.uniform(0.1, 0.5),
                    "sharpe_ratio": random.uniform(-0.5, 2.5),
                    "debt_to_equity": random.uniform(0.1, 2.0)
                }
                
                # News headlines from Tavily integration
                self.news_headlines = [
                    {
                        "title": f"{symbol} Reports Record Quarterly Results",
                        "source": "MarketWatch",
                        "date": (datetime.now() - timedelta(days=random.randint(1, 7))).strftime("%Y-%m-%d"),
                        "url": f"https://marketwatch.com/symbol/{symbol.lower()}/news/earnings"
                    },
                    {
                        "title": f"Analysts Upgrade {symbol} Following Strong Performance",
                        "source": "Seeking Alpha",
                        "date": (datetime.now() - timedelta(days=random.randint(1, 7))).strftime("%Y-%m-%d"),
                        "url": f"https://seekingalpha.com/symbol/{symbol.lower()}/news/upgrade"
                    },
                    {
                        "title": f"{symbol} Announces New Product Launch",
                        "source": "Bloomberg",
                        "date": (datetime.now() - timedelta(days=random.randint(1, 7))).strftime("%Y-%m-%d"),
                        "url": f"https://bloomberg.com/news/{symbol.lower()}/product-launch"
                    },
                    {
                        "title": f"Market Volatility Impacts {symbol} Stock Price",
                        "source": "CNBC",
                        "date": (datetime.now() - timedelta(days=random.randint(1, 7))).strftime("%Y-%m-%d"),
                        "url": f"https://cnbc.com/stock/{symbol.lower()}/news/volatility"
                    },
                    {
                        "title": f"Regulatory Changes Could Affect {symbol}'s Business Model",
                        "source": "Wall Street Journal",
                        "date": (datetime.now() - timedelta(days=random.randint(1, 7))).strftime("%Y-%m-%d"),
                        "url": f"https://wsj.com/articles/{symbol.lower()}-regulatory-impact"
                    }
                ]
                
                # Risk keywords from news analysis
                self.risk_keywords = {
                    "high_risk": random.sample(["lawsuit", "investigation", "downgrade", "recall", "fraud", "bankruptcy", "fine", "penalty"], k=random.randint(0, 3)),
                    "moderate_risk": random.sample(["volatility", "competition", "uncertainty", "challenge", "regulation", "decline", "pressure", "concern"], k=random.randint(1, 4)),
                    "positive": random.sample(["growth", "upgrade", "innovation", "profit", "expansion", "partnership", "leadership", "dividend"], k=random.randint(2, 5))
                }
                
                # News sentiment data
                sentiment_score = random.uniform(0.2, 0.8)
                self.news_sentiment = {
                    "score": sentiment_score,
                    "sentiment": "positive" if sentiment_score > 0.6 else "neutral" if sentiment_score > 0.4 else "negative",
                    "confidence": random.uniform(0.6, 0.95)
                }
                
                # Sentiment description for the UI
                sentiment_term = "positive" if sentiment_score > 0.6 else "mixed" if sentiment_score > 0.4 else "negative"
                self.sentiment_description = f"Recent news coverage for {symbol} shows {sentiment_term} sentiment with a score of {sentiment_score:.2f}. "
                
                if self.risk_keywords["high_risk"]:
                    self.sentiment_description += f"Concerning keywords like '{', '.join(self.risk_keywords['high_risk'])}' were detected in recent articles. "
                
                if self.risk_keywords["positive"]:
                    self.sentiment_description += f"However, positive indicators such as '{', '.join(self.risk_keywords['positive'][:3])}' were also present in the coverage. "
                
                self.sentiment_description += f"The overall news sentiment suggests {'increased vigilance' if sentiment_score < 0.4 else 'cautious optimism' if sentiment_score < 0.7 else 'favorable conditions'} for investors considering {symbol}."
                
                # Original GPT insights
                risk_level = "high" if self.composite_risk_score > 0.7 else "moderate" if self.composite_risk_score > 0.4 else "low"
                
                # Generate realistic volatility metrics
                historical_volatility = round(random.uniform(0.01, 0.04), 3)  # Daily volatility 1-4%
                annualized_volatility = round(historical_volatility * np.sqrt(252), 3)  # Annualized volatility 15-60%
                max_drawdown = round(random.uniform(15, 40), 1)  # Realistic max drawdown 15-40%
                sharpe = self.metrics['sharpe_ratio']  # Use existing sharpe ratio
                
                # Determine realistic descriptions
                vol_trend = random.choice(["stable", "increasing", "decreasing"])
                
                # Use the overall risk score to determine volatility risk level (same as backend logic)
                # Convert composite_risk_score from 0-1 to 0-100 scale for threshold comparisons
                risk_score_100 = self.composite_risk_score * 100
                if risk_score_100 > 75:
                    vol_risk_level = "very high"
                elif risk_score_100 > 55:
                    vol_risk_level = "high"
                elif risk_score_100 > 30:
                    vol_risk_level = "moderate"
                else:
                    vol_risk_level = "low"
                    
                return_quality = "poor" if sharpe < 0.5 else "adequate" if sharpe < 1.0 else "good"
                
                # Fixed volatility statement with realistic values
                vol_statement = f"Volatility: The stock has {annualized_volatility*100:.1f}% annualized volatility with a maximum drawdown of {max_drawdown:.1f}%. "
                vol_statement += f"Volatility has been {vol_trend} recently (not increasing or decreasing significantly). "
                vol_statement += f"The risk-adjusted return (Sharpe ratio: {sharpe:.2f}) is {return_quality}. "
                vol_statement += f"Based on these metrics and the overall risk assessment, volatility represents a {vol_risk_level} risk component."
                
                # Complete insights with the fixed volatility statement
                self.gpt_insights = f"{symbol} presents a {risk_level} risk profile based on our comprehensive analysis. {vol_statement} The stock shows {'elevated' if self.risk_factors['Market Risk'] > 0.6 else 'moderate' if self.risk_factors['Market Risk'] > 0.4 else 'low'} market risk with beta of {self.metrics['beta']:.2f}, suggesting {'higher' if self.metrics['beta'] > 1.2 else 'similar' if self.metrics['beta'] > 0.8 else 'lower'} volatility than the overall market. Financial leverage is {'concerning' if self.risk_factors['Financial Leverage Risk'] > 0.6 else 'manageable' if self.risk_factors['Financial Leverage Risk'] > 0.4 else 'very low'} with a debt-to-equity ratio of {self.metrics['debt_to_equity']:.2f}."
                
                # Add news and sentiment insights
                self.gpt_insights += f"\n\nRecent news analysis reveals {sentiment_term} sentiment trends. {random.randint(60, 90)}% of articles published in the past week express {'optimism' if sentiment_score > 0.6 else 'caution' if sentiment_score > 0.4 else 'concern'} about {symbol}'s prospects. {'Several high-risk keywords were identified, suggesting potential challenges ahead.' if self.risk_keywords['high_risk'] else 'Few serious risk factors were identified in recent coverage.'} Overall, the media narrative {'supports' if sentiment_score > 0.5 else 'adds caution to'} the quantitative risk assessment."
        
        return MockRiskData(symbol)
    
    @staticmethod
    def get_risk_assessment(symbol: str = None):
        """Wrapper for assess_risk to match the API expected by the mock RiskAssessmentAgent."""
        if not symbol:
            symbol = "AAPL"
        return MockDataProvider.assess_risk(symbol)
    
    @staticmethod
    def get_web_research(query: str) -> List[Dict[str, str]]:
        """Wrapper for scrape to match the API expected by the mock TavilyScraper."""
        return MockDataProvider.scrape(query)
    
    @staticmethod
    def scrape(query: str) -> List[Dict[str, str]]:
        """Mock Tavily web scraping results."""
        topics = [
            "Quarterly Earnings Report",
            "Industry Analysis",
            "CEO Interview",
            "Market Outlook",
            "Analyst Ratings",
            "Technical Analysis",
            "Competitor Comparison",
            "Growth Prospects"
        ]
        
        results = []
        for i in range(min(5, len(topics))):
            symbol = query.split()[0]  # Extract the stock symbol from query
            topic = topics[i]
            
            results.append({
                "title": f"{symbol} {topic}: What Investors Need to Know",
                "content": f"This article discusses {symbol}'s {topic.lower()}. " + 
                          f"According to recent analysis, {symbol} has shown {'promising' if random.random() > 0.3 else 'concerning'} trends in this area. " +
                          f"Experts {'believe' if random.random() > 0.5 else 'question whether'} the company will {'continue to grow' if random.random() > 0.4 else 'face challenges'} in the coming quarters. " +
                          f"Several key factors to consider include market conditions, management decisions, and competitive landscape. " +
                          "This information should be considered as part of a comprehensive investment strategy.",
                "source": random.choice(["MarketWatch", "Seeking Alpha", "The Motley Fool", "Investopedia", "Forbes", "Bloomberg"]),
                "url": f"https://example.com/{symbol.lower()}/{topic.lower().replace(' ', '-')}"
            })
        
        return results


# Demo usage
if __name__ == "__main__":
    mock = MockDataProvider()
    print("Sample Financial Snapshot:")
    print(mock.get_financial_snapshot("AAPL")['data']['market_cap'])
    
    print("\nSample News:")
    news = mock.get_news("AAPL")
    for item in news['data'][:2]:
        print(f"- {item['title']} ({item['source']})")
    
    print("\nSample Sentiment:")
    sentiment = mock.get_news_sentiment("AAPL")
    print(f"Score: {sentiment.sentiment_score}, Confidence: {sentiment.confidence}")
