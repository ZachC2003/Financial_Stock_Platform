"""
Sentiment Analyzer Module

This module provides sentiment analysis for financial news articles using pre-trained models.
It uses a natural language processing approach to evaluate the sentiment of text.
"""

import re
import logging
from typing import Dict, Any, List, Union, Optional
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from textblob import TextBlob

# Set up logging
logger = logging.getLogger(__name__)

# Financial-specific terms with their sentiment scores
FINANCIAL_TERMS = {
    # Positive terms
    "growth": 0.4,
    "profit": 0.6,
    "bullish": 0.7,
    "outperform": 0.6,
    "beat": 0.5,
    "upgrade": 0.65,
    "strong": 0.5,
    "positive": 0.6,
    "opportunity": 0.5,
    "rise": 0.4,
    "gain": 0.5,
    "improve": 0.4,
    "increase": 0.3,
    "exceed": 0.5,
    "upside": 0.5,
    "recovery": 0.4,
    "rally": 0.6,
    "breakthrough": 0.6,
    
    # Negative terms
    "loss": -0.6,
    "bearish": -0.7,
    "underperform": -0.6,
    "miss": -0.5,
    "downgrade": -0.65,
    "weak": -0.5,
    "negative": -0.6,
    "risk": -0.5,
    "fall": -0.4,
    "drop": -0.5,
    "decline": -0.4,
    "decrease": -0.3,
    "disappoint": -0.6,
    "downside": -0.5,
    "downturn": -0.6,
    "crash": -0.8,
    "layoffs": -0.7,
    "lawsuit": -0.6,
    "investigation": -0.6,
    "recall": -0.7,
    "fine": -0.5,
    "penalty": -0.5,
    "default": -0.7,
    "bankruptcy": -0.9,
    "debt": -0.4,
    "inflation": -0.4,
    "recession": -0.7,
    "volatility": -0.4,
}


class SentimentAnalyzer:
    """
    A class for analyzing sentiment in financial news text.
    Uses NLTK's VADER sentiment analyzer with financial domain adjustments.
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer"""
        try:
            # Download VADER lexicon if not already downloaded
            try:
                nltk.data.find('vader_lexicon')
            except LookupError:
                nltk.download('vader_lexicon', quiet=True)
                
            # Initialize VADER sentiment analyzer
            self.vader = SentimentIntensityAnalyzer()
            
            # Add financial domain-specific terms
            for term, score in FINANCIAL_TERMS.items():
                self.vader.lexicon[term] = score
                
            logger.info("Sentiment analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {str(e)}")
            self.vader = None
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of a text using VADER with financial adjustments.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            Dict[str, Any]: Dictionary containing sentiment scores and classification
        """
        if not text or not self.vader:
            # Return neutral if no text or analyzer
            return {
                "positive": 0.33,
                "neutral": 0.34,
                "negative": 0.33,
                "compound": 0.0,
                "sentiment": "neutral",
                "confidence": 0.5
            }
        
        try:
            # Clean the text
            cleaned_text = self._clean_text(text)
            
            # Get VADER sentiment scores
            vader_scores = self.vader.polarity_scores(cleaned_text)
            
            # Get TextBlob sentiment as secondary measure
            blob = TextBlob(cleaned_text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            # Combine VADER and TextBlob scores
            compound = vader_scores['compound']
            
            # Determine sentiment label
            if compound >= 0.25:
                sentiment = "positive"
            elif compound <= -0.25:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            # Calculate confidence based on TextBlob subjectivity and VADER compound score
            confidence = (abs(compound) * 0.7) + (textblob_subjectivity * 0.3)
            
            return {
                "positive": vader_scores['pos'],
                "neutral": vader_scores['neu'],
                "negative": vader_scores['neg'],
                "compound": compound,
                "sentiment": sentiment,
                "confidence": min(abs(confidence), 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {
                "positive": 0.33,
                "neutral": 0.34,
                "negative": 0.33,
                "compound": 0.0,
                "sentiment": "neutral",
                "confidence": 0.5
            }
    
    def _clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis"""
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_news_batch(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of news items.
        
        Args:
            news_items (List[Dict[str, Any]]): List of news item dictionaries
            
        Returns:
            List[Dict[str, Any]]: The same list with sentiment analysis added
        """
        if not news_items:
            return []
            
        try:
            for item in news_items:
                # Create text to analyze from title and description/summary if available
                text_to_analyze = item.get('title', '')
                if item.get('description'):
                    text_to_analyze += ' ' + item['description']
                elif item.get('summary'):
                    text_to_analyze += ' ' + item['summary']
                
                # Analyze sentiment
                sentiment_results = self.analyze_text(text_to_analyze)
                
                # Add sentiment to the news item
                item['sentiment'] = sentiment_results['sentiment']
                item['sentiment_scores'] = {
                    'positive': sentiment_results['positive'],
                    'neutral': sentiment_results['neutral'],
                    'negative': sentiment_results['negative'],
                    'compound': sentiment_results['compound']
                }
                item['sentiment_confidence'] = sentiment_results['confidence']
            
            # Calculate aggregate sentiment
            aggregate = self._calculate_aggregate_sentiment(news_items)
            
            return news_items, aggregate
            
        except Exception as e:
            logger.error(f"Error batch analyzing news: {str(e)}")
            return news_items
    
    def _calculate_aggregate_sentiment(self, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate sentiment for a list of news items"""
        if not news_items:
            return {
                "sentiment_summary": "neutral", 
                "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
                "average_compound": 0.0
            }
            
        try:
            # Count sentiments
            sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
            compound_sum = 0
            
            for item in news_items:
                sentiment = item.get('sentiment', 'neutral')
                sentiment_counts[sentiment] += 1
                compound_sum += item.get('sentiment_scores', {}).get('compound', 0)
            
            total = len(news_items)
            average_compound = compound_sum / total if total > 0 else 0
            
            # Calculate percentages
            sentiment_distribution = {
                k: round(v / total * 100) if total > 0 else 0 
                for k, v in sentiment_counts.items()
            }
            
            # Determine overall sentiment
            if average_compound >= 0.25:
                overall = "bullish"
            elif average_compound <= -0.25:
                overall = "bearish"
            else:
                overall = "neutral"
                
            return {
                "sentiment_summary": overall,
                "sentiment_distribution": sentiment_distribution,
                "average_compound": average_compound
            }
            
        except Exception as e:
            logger.error(f"Error calculating aggregate sentiment: {str(e)}")
            return {
                "sentiment_summary": "neutral", 
                "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
                "average_compound": 0.0
            }


# Singleton instance
_sentiment_analyzer = None

def get_sentiment_analyzer():
    """Get or create a singleton instance of the sentiment analyzer"""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer
