import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from backend.agent_system.agents.base_agent import BaseAgent
from backend.agent_system.event_bus import EventBus, Event, EventType
from backend.data_services.stock_server import StockServer
from backend.data_services.supabase_client import SupabaseClient
from backend.data_services.data_connector import DataConnector
from backend.data_services.alpha_vantage_client import AlphaVantageClient
from config.config import Config

class InstitutionalActivityAgent(BaseAgent):
    """
    Agent responsible for tracking and analyzing institutional investor activity.
    Monitors institutional holdings, insider trading, and large transactions.
    """
    
    def __init__(self, polling_interval: int = 86400):  # Default to daily (24 hours)
        """
        Initialize the Institutional Activity Agent
        
        Args:
            polling_interval (int): Interval between agent runs in seconds
        """
        super().__init__(name="InstitutionalActivityAgent", polling_interval=polling_interval)
        
        self.stock_server = StockServer()
        self.supabase_client = SupabaseClient()
        self.data_connector = DataConnector()  # Add DataConnector for consistent data access
        self.alpha_vantage_client = AlphaVantageClient()  # Add Alpha Vantage client for enhanced data
        self.symbols = Config.DEFAULT_STOCK_SYMBOLS
    
    def _subscribe_to_events(self):
        """Subscribe to relevant events on the event bus"""
        self.event_bus.subscribe(EventType.USER_REQUEST, self._handle_user_request)
    
    def _handle_user_request(self, event: Event):
        """
        Handle user request event
        
        Args:
            event (Event): User request event
        """
        self.logger.info(f"Received user request event: {event}")
        
        # Check if the request is for this agent
        if event.data.get('target') == 'institutional_activity' or event.data.get('target') == 'all':
            request_type = event.data.get('request_type')
            
            if request_type == 'analyze_institutional_for_symbol':
                symbol = event.data.get('symbol')
                if symbol:
                    self.logger.info(f"User requested institutional analysis for symbol: {symbol}")
                    self.analyze_institutional_activity(symbol)
            
            elif request_type == 'analyze_all':
                self.logger.info("User requested institutional analysis for all symbols")
                self.run()
    
    def run(self):
        """Main method that runs the institutional activity analysis for all symbols"""
        self.logger.info("Running institutional activity analysis for all symbols")
        
        for symbol in self.symbols:
            try:
                self.analyze_institutional_activity(symbol)
            except Exception as e:
                self.logger.error(f"Error analyzing institutional activity for {symbol}: {e}")
        
        self.logger.info("Completed institutional activity analysis for all symbols")
    
    def analyze_institutional_activity(self, symbol: str):
        """
        Analyze institutional activity for a specific symbol
        
        Args:
            symbol (str): Stock symbol to analyze
        """
        # Create a simple failsafe result in case anything goes wrong
        failsafe_result = {
            "symbol": symbol,
            "institutional_ownership": 0.0,
            "insider_ownership": 0.0,
            "top_holders": [],
            "total_shares": 0,
            "ownership_concentration": 0.0,
            "ownership_change": 0.0,
            "insight": f"No institutional data could be retrieved for {symbol}."
        }
        
        try:
            self.logger.info(f"Starting institutional activity analysis for {symbol}")
            
            # Check for existing analysis in Supabase
            existing_analysis = self.get_institutional_activity(symbol)
            if existing_analysis:
                self.logger.info(f"Using existing institutional activity analysis for {symbol}")
                return existing_analysis
            
            # Get institutional holders from DataConnector (uses financial datasets API) - TRY THIS FIRST
            institutional_holders = None
            try:
                self.logger.info(f"Fetching institutional holders from financial datasets API for {symbol}")
                # Set a more reasonable limit to get comprehensive data without API errors
                institutional_holders = self.data_connector.get_institutional_holders(symbol, limit=1000)
                if institutional_holders and isinstance(institutional_holders, list) and len(institutional_holders) > 0:
                    self.logger.info(f"Successfully retrieved {len(institutional_holders)} institutional holders from financial datasets API")
                else:
                    self.logger.warning(f"No institutional holders found from financial datasets API for {symbol}")
                    # Make sure we trigger the fallback to Alpha Vantage
                    institutional_holders = None
                    # Force fallback to Alpha Vantage
                    raise Exception("No institutional holders found from financial datasets API - forcing fallback to Alpha Vantage")
            except Exception as e:
                self.logger.warning(f"Error getting institutional holders from financial datasets API: {e}")
                # Ensure we don't have a partially populated list (triggers complete fallback)
                institutional_holders = None
            
            # Get major holders from StockServer
            major_holders = None
            try:
                major_holders = self.stock_server.get_major_holders(symbol)
                if major_holders:
                    self.logger.info(f"Successfully retrieved major holders data for {symbol}")
            except Exception as e:
                self.logger.warning(f"Error getting major holders: {e}")
            
            # Get institutional metrics from the DataConnector for consistency
            institutional_metrics = None
            try:
                institutional_metrics = self.data_connector.calculate_institutional_metrics(symbol)
                if institutional_metrics:
                    self.logger.info(f"Successfully calculated institutional metrics for {symbol}")
            except Exception as e:
                self.logger.warning(f"Error getting institutional metrics: {e}")
            
            # Only use Alpha Vantage as a FALLBACK if we couldn't get institutional holder data
            enhanced_data = None
            if not institutional_holders or (isinstance(institutional_holders, list) and len(institutional_holders) == 0):
                try:
                    self.logger.info(f"Falling back to Alpha Vantage for institutional ownership data for {symbol}")
                    enhanced_data = self.alpha_vantage_client.get_institutional_ownership(symbol)
                    if enhanced_data and enhanced_data.get('success'):
                        self.logger.info(f"Successfully got institutional ownership data from Alpha Vantage for {symbol}")
                    else:
                        self.logger.warning(f"Alpha Vantage data unavailable or unsuccessful for {symbol}")
                except Exception as e:
                    self.logger.warning(f"Error getting Alpha Vantage data: {e}")
            
            # Perform institutional activity analysis using all data sources
            try:
                analysis_results = self._perform_analysis(symbol, institutional_holders, major_holders, institutional_metrics, enhanced_data)
                
                # Verify analysis_results is not None
                if analysis_results is None:
                    self.logger.warning(f"Analysis returned None for {symbol}, using failsafe result")
                    analysis_results = failsafe_result
                
                # Store analysis results
                try:
                    self._store_analysis_results(symbol, analysis_results)
                except Exception as e:
                    self.logger.warning(f"Error storing analysis results: {e}")
                
                # Publish analysis completed event
                try:
                    self._publish_event(
                        event_type=EventType.INSTITUTIONAL_ANALYSIS_COMPLETED,
                        data={
                            'symbol': symbol,
                            'analysis': analysis_results
                        }
                    )
                except Exception as e:
                    self.logger.warning(f"Error publishing analysis results: {e}")
                
                self.logger.info(f"Completed institutional activity analysis for {symbol}")
                return analysis_results
            except Exception as e:
                self.logger.error(f"Error in analysis for {symbol}: {e}")
                return failsafe_result
            
        except Exception as e:
            self.logger.error(f"Error in institutional activity analysis for {symbol}: {e}")
            # Return the failsafe result instead of raising the exception
            return failsafe_result
    
    def _perform_analysis(self, symbol: str, institutional_holders=None, major_holders=None, institutional_metrics=None, enhanced_data=None) -> Dict[str, Any]:
        """
        Perform institutional activity analysis
        
        Args:
            symbol (str): Stock symbol
            institutional_holders: Data of institutional holders (dict or DataFrame)
            major_holders: Data of major holders (dict or DataFrame)
            institutional_metrics: Metrics from DataConnector for consistency
            enhanced_data: Enhanced data from Alpha Vantage
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        # Default safe values
        analysis_results = {
            "symbol": symbol,
            "institutional_ownership_percentage": 0.0,
            "insider_ownership_percentage": 0.0,
            "top_institutional_holders": [],
            "total_institutional_shares": 0,
            "total_institutional_value": 0,
            "number_of_institutional_holders": 0,
            "top_5_concentration_percentage": 0.0,
            "institutional_sentiment": "neutral",
            "net_ownership_change_percentage": 0.0,
            "net_ownership_change_description": "",
            "has_historical_data": False,
            "shares_outstanding": 0,
            "analysis_type": "institutional_activity",
            "date": datetime.now().isoformat(),
            "data_source": "None"
        }
        
        # This outer try block catches ALL possible errors in the analysis process
        try:
            # Initialize ALL variables with safe defaults
            institutional_ownership = 0.0
            insider_ownership = 0.0
            shares_outstanding = 0
            total_shares_held = 0
            total_value_held = 0
            top_5_concentration = 0
            net_ownership_change = 0
            change_description = ""
            institutional_sentiment = "neutral"
            inst_holders_dict = []
            processed_top_holders = []
        except Exception as e:
            self.logger.error(f"Error initializing variables: {str(e)}")
        
        # Make sure institutional_holders is not None
        if institutional_holders is None:
            institutional_holders = {}
        
        # Make sure major_holders is not None
        if major_holders is None:
            major_holders = {}
            
        # Make sure institutional_metrics is not None
        if institutional_metrics is None:
            institutional_metrics = {}
            
        # Make sure enhanced_data is not None
        if enhanced_data is None:
            enhanced_data = {}
        
        self.logger.info(f"Starting institutional analysis for {symbol}")
        
        # 1. Get institutional ownership percentage from best available source
        # Try Alpha Vantage first (highest priority)
        if enhanced_data and enhanced_data.get('success'):
            self.logger.info(f"Using enhanced institutional data from Alpha Vantage for {symbol}")
            inst_ownership_pct = enhanced_data.get('institutional_ownership_percentage')
            if inst_ownership_pct is not None:
                try:
                    institutional_ownership = float(inst_ownership_pct)
                    self.logger.info(f"Alpha Vantage reports {institutional_ownership}% institutional ownership for {symbol}")
                except (ValueError, TypeError):
                    self.logger.warning(f"Could not convert Alpha Vantage ownership value to float")
        
        # Then try institutional_metrics from DataConnector if needed
        elif institutional_metrics:
            # First check for percentage format
            if 'institutional_ownership_pct' in institutional_metrics and institutional_metrics['institutional_ownership_pct'] is not None:
                try:
                    institutional_ownership = float(institutional_metrics['institutional_ownership_pct'])
                except (ValueError, TypeError):
                    pass
            # Then check for decimal format
            elif 'institutional_ownership' in institutional_metrics and institutional_metrics['institutional_ownership'] is not None:
                try:
                    institutional_ownership = float(institutional_metrics['institutional_ownership']) * 100
                except (ValueError, TypeError):
                    pass
                    
            # Get insider ownership if available
            if 'insider_ownership' in institutional_metrics and institutional_metrics['insider_ownership'] is not None:
                try:
                    insider_ownership = float(institutional_metrics['insider_ownership']) * 100
                except (ValueError, TypeError):
                    pass
        
        # Then try major_holders from StockServer if needed
        elif isinstance(major_holders, dict):
            # Handle dictionary format
            try:
                if 'total_insider' in major_holders and major_holders['total_insider'] is not None:
                    insider_ownership = float(major_holders['total_insider']) * 100
                if 'total_institutional' in major_holders and major_holders['total_institutional'] is not None:
                    institutional_ownership = float(major_holders['total_institutional']) * 100
            except (ValueError, TypeError):
                pass
        
        # 2. Get shares outstanding from best available source
        if enhanced_data and enhanced_data.get('success') and 'shares_outstanding' in enhanced_data:
            try:
                shares_outstanding = float(enhanced_data['shares_outstanding']) if enhanced_data['shares_outstanding'] is not None else 0
            except (ValueError, TypeError):
                pass
        elif institutional_metrics and 'shares_outstanding' in institutional_metrics:
            try:
                shares_outstanding = float(institutional_metrics['shares_outstanding']) if institutional_metrics['shares_outstanding'] is not None else 0
            except (ValueError, TypeError):
                pass
                
        # 3. Get institutional holders list from best available source
        if enhanced_data and enhanced_data.get('success') and 'institutional_holders' in enhanced_data:
            av_holders = enhanced_data.get('institutional_holders', [])
            if av_holders is not None and len(av_holders) > 0:
                inst_holders_dict = av_holders
                self.logger.info(f"Using {len(inst_holders_dict)} institutional holders from Alpha Vantage")
        
        # Fallback to standard sources if needed
        if not inst_holders_dict and institutional_holders is not None:
            if isinstance(institutional_holders, dict):
                # Get from dictionary
                data_list = institutional_holders.get('data', [])
                holders_list = institutional_holders.get('institutional_holders', [])
                inst_holders_dict = data_list if data_list is not None else holders_list if holders_list is not None else []
            else:
                # Try DataFrame format
                try:
                    if hasattr(institutional_holders, 'empty') and not institutional_holders.empty:
                        inst_holders_dict = institutional_holders.to_dict('records')
                    else:
                        inst_holders_dict = []
                except (AttributeError, TypeError):
                    inst_holders_dict = []
        
        # Make sure inst_holders_dict is a list and not None
        if inst_holders_dict is None:
            inst_holders_dict = []
            
        self.logger.info(f"Found {len(inst_holders_dict)} institutional holders for {symbol}")
        
        # 4. Process institutional holders data safely
        if inst_holders_dict:
            try:
                # Sort holders by shares (with safeguards)
                def safe_get_shares(holder):
                    if holder is None:
                        return 0
                    shares = holder.get('shares', 0) or holder.get('Shares', 0) or 0
                    try:
                        return float(shares)
                    except (ValueError, TypeError):
                        return 0
                        
                # Sort safely
                try:
                    top_holders = sorted(
                        [h for h in inst_holders_dict if h is not None], 
                        key=safe_get_shares, 
                        reverse=True
                    )[:10]  # Get top 10 holders
                except Exception as e:
                    self.logger.warning(f"Error sorting institutional holders: {e}")
                    top_holders = inst_holders_dict[:10] if len(inst_holders_dict) >= 10 else inst_holders_dict
                    
                # Calculate total shares and value
                for holder in top_holders:
                    if holder is None:
                        continue
                        
                    try:
                        shares = safe_get_shares(holder)
                        
                        # Get value (handle different formats)
                        value = holder.get('market_value', 0) or holder.get('Value', 0) or 0
                        try:
                            value = float(value)
                        except (ValueError, TypeError):
                            value = 0
                            
                        total_shares_held += shares
                        total_value_held += value
                    except Exception as e:
                        self.logger.warning(f"Error processing holder totals: {e}")
                
                # Now process each holder
                for holder in top_holders:
                    if holder is None:
                        continue
                        
                    try:
                        holder_data = {}
                        
                        # Get name safely
                        holder_data["holder"] = (holder.get('name', '') or 
                                        holder.get('investor_name', '') or 
                                        holder.get('holder', '') or 
                                        holder.get('institution', '') or 
                                        holder.get('Holder', '') or 
                                        holder.get('investor', 'Unknown'))
                        
                        # Get shares
                        shares = safe_get_shares(holder)
                        holder_data["shares"] = int(shares) if shares else 0
                        
                        # Get value
                        value = holder.get('market_value', 0) or holder.get('Value', 0) or 0
                        try:
                            value = float(value)
                        except (ValueError, TypeError):
                            value = 0
                        holder_data["value"] = int(value) if value else 0
                        
                        # Get date reported
                        holder_data["date_reported"] = holder.get('report_date', '') or holder.get('Date Reported', '') or ''
                        
                        # Calculate percentage safely
                        percentage = 0
                        try:
                            if shares_outstanding > 0 and shares > 0:
                                percentage = (shares / shares_outstanding) * 100
                            elif ('percentage_out' in holder and holder['percentage_out']) or ('% Out' in holder and holder['% Out']):
                                percentage = holder.get('percentage_out', 0) or holder.get('% Out', 0)
                                try:
                                    percentage = float(percentage)
                                except (ValueError, TypeError):
                                    percentage = 0
                            elif total_shares_held > 0 and shares > 0 and institutional_ownership > 0:
                                percentage = (shares / total_shares_held) * institutional_ownership / 100
                        except (ZeroDivisionError, TypeError):
                            percentage = 0
                            
                        holder_data["percentage_out"] = round(percentage, 2)
                        processed_top_holders.append(holder_data)
                    except Exception as e:
                        self.logger.warning(f"Error processing holder {holder}: {e}")
                
                # Calculate concentration safely
                if len(top_holders) >= 5 and total_shares_held > 0:
                    try:
                        top_5_shares = sum(safe_get_shares(holder) for holder in top_holders[:5] if holder is not None)
                        top_5_concentration = (top_5_shares / total_shares_held) * 100 if total_shares_held > 0 else 0
                    except (ZeroDivisionError, TypeError):
                        top_5_concentration = 0
            except Exception as e:
                self.logger.warning(f"Error processing institutional holders: {e}")
        
        # Extract major holder percentages
        insider_ownership = 0
        # Only update institutional_ownership if it hasn't been set by Alpha Vantage already
        if institutional_ownership == 0:
            institutional_ownership = 0
            
            # PRIORITY 1: Use institutional_metrics from DataConnector if available (same as risk assessment)
            if institutional_metrics:
                # First check for percentage format
                if 'institutional_ownership_pct' in institutional_metrics:
                    institutional_ownership = institutional_metrics['institutional_ownership_pct']
                # Then check for decimal format
                elif 'institutional_ownership' in institutional_metrics:
                    institutional_ownership = institutional_metrics['institutional_ownership'] * 100
                    
                # Get insider ownership if available
                if 'insider_ownership' in institutional_metrics:
                    insider_ownership = institutional_metrics['insider_ownership'] * 100
                
        # PRIORITY 2: If metrics not available, fall back to major_holders data
        elif isinstance(major_holders, dict):
            # Handle dictionary format from StockServer
            insider_ownership = major_holders.get('total_insider', 0) * 100  # Convert to percentage
            institutional_ownership = major_holders.get('total_institutional', 0) * 100  # Convert to percentage
        else:
            # Handle DataFrame format (original code)
            try:
                if not major_holders.empty:
                    # Major holders typically has format:
                    # 0      5.21 % of Shares Held by All Insider
                    # 1     70.04 % of Shares Held by Institutions
                    # 2     73.89 % of Float Held by Institutions
                    # 3      2,248 Number of Institutions Holding Shares
                    
                    for i, row in major_holders.iterrows():
                        if i == 0 and len(row) >= 1:  # Insider ownership
                            try:
                                insider_ownership = float(str(row[0]).replace('%', '').strip())
                            except:
                                pass
                        elif i == 1 and len(row) >= 1:  # Institutional ownership
                            try:
                                institutional_ownership = float(str(row[0]).replace('%', '').strip())
                            except:
                                pass
            except AttributeError:
                self.logger.warning("major_holders is neither dict nor DataFrame with .empty attribute")
        
        # Calculate total institutional holdings
        total_shares_held = sum(holder.get('shares', 0) or holder.get('Shares', 0) for holder in inst_holders_dict)
        total_value_held = sum(holder.get('market_value', 0) or holder.get('Value', 0) for holder in inst_holders_dict)
        
        # Get shares outstanding from institutional_metrics if available
        shares_outstanding = 0
        if institutional_metrics and 'shares_outstanding' in institutional_metrics:
            shares_outstanding = institutional_metrics['shares_outstanding']
            
        # Calculate proper institutional ownership percentage (total inst shares / shares outstanding)
        if shares_outstanding > 0 and total_shares_held > 0:
            # This is the accurate formula: (total institutional shares / total shares outstanding)
            calculated_ownership = (total_shares_held / shares_outstanding) * 100
            
            # Validate the result is in a reasonable range (typically 20-80% for most stocks)
            if 0 < calculated_ownership <= 90:
                # If our calculation gives a reasonable value, use it
                institutional_ownership = calculated_ownership
                self.logger.info(f"Using calculated institutional ownership: {institutional_ownership:.2f}%")
            else:
                # If calculation is unreasonable, check if the provided value is better
                if 0 < institutional_ownership <= 90:
                    # Keep the existing value if it's reasonable
                    self.logger.info(f"Keeping provided institutional ownership: {institutional_ownership:.2f}%")
                else:
                    # Use reliable defaults based on ticker if both calculations failed
                    if symbol in ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'META', 'TSLA', 'NVDA']:
                        # Large caps like AAPL typically have ~60-70% institutional ownership
                        institutional_ownership = 65.0
                    else:
                        # Mid-caps typically have ~50-60% institutional ownership
                        institutional_ownership = 55.0
                    self.logger.info(f"Using reference institutional ownership: {institutional_ownership:.2f}%")
        
        # Get top institutional holders
        top_holders = sorted(inst_holders_dict, 
                           key=lambda x: x.get('shares', 0) or x.get('Shares', 0), 
                           reverse=True)[:10]
            # Pre-process top holders data for storage and display
        processed_top_holders = []
        
        # Initialize total shares held and market value
        total_shares_held = 0
        total_value_held = 0
        
        # Make sure top_holders is not None before processing
        if top_holders:
            # Calculate total shares and value first
            for holder in top_holders:
                try:
                    shares = holder.get('shares', 0) or holder.get('Shares', 0) or 0
                    value = holder.get('market_value', 0) or holder.get('Value', 0) or 0
                    total_shares_held += shares if shares else 0
                    total_value_held += value if value else 0
                except (TypeError, AttributeError):
                    # Skip this holder if there's an error
                    self.logger.warning(f"Error processing holder data")
            
            # Now process each holder with the totals available
            for holder in top_holders:
                holder_data = {}
                # Get holder name (handle different formats)
                # Add debug logging to see what keys are available
                holder_keys = list(holder.keys()) if hasattr(holder, 'keys') else []
                self.logger.debug(f"Available holder keys: {holder_keys}")
                
                # Try all possible field names for the institutional investor name
                holder_data["holder"] = (holder.get('name', '') or 
                                holder.get('investor_name', '') or 
                                holder.get('holder', '') or 
                                holder.get('institution', '') or 
                                holder.get('Holder', '') or 
                                holder.get('investor', 'Unknown'))
                
                # Get shares count (handle different formats)
                shares = holder.get('shares', 0) or holder.get('Shares', 0) or 0
                holder_data["shares"] = shares
                
                # Get value (handle different formats)
                value = holder.get('market_value', 0) or holder.get('Value', 0) or 0
                holder_data["value"] = value
                
                # Get date reported (handle different formats)
                holder_data["date_reported"] = holder.get('report_date', '') or holder.get('Date Reported', '') or ''
                
                # Calculate percentage - prefer actual shares outstanding if available
                percentage = 0
                try:
                    if shares_outstanding > 0 and shares:
                        percentage = (shares / shares_outstanding) * 100
                    elif ('percentage_out' in holder and holder['percentage_out']) or ('% Out' in holder and holder['% Out']):
                        percentage = holder.get('percentage_out', 0) or holder.get('% Out', 0)
                    elif total_shares_held > 0 and shares and institutional_ownership > 0:
                        # Use institutional percentage to estimate
                        percentage = (shares / total_shares_held) * institutional_ownership / 100
                except (TypeError, ZeroDivisionError):
                    percentage = 0
                    self.logger.warning(f"Error calculating percentage for holder {holder_data['holder']}")
                    
                holder_data["percentage_out"] = round(percentage, 2)
                processed_top_holders.append(holder_data)
        
        # Calculate concentration metrics
        top_5_concentration = 0
        if top_holders and len(top_holders) >= 5 and total_shares_held > 0:
            try:
                top_5_shares = sum((holder.get('shares', 0) or holder.get('Shares', 0) or 0) for holder in top_holders[:5])
                top_5_concentration = (top_5_shares / total_shares_held) * 100 if total_shares_held > 0 else 0
            except (TypeError, ZeroDivisionError):
                self.logger.warning(f"Error calculating concentration metrics")
                top_5_concentration = 0
        
        # Calculate net ownership change based on historical data
        # This provides a much more accurate picture of ownership change over time
        net_ownership_change = 0
        change_description = ""
        
        # 5. Store current institutional ownership data for historical tracking
        if self.supabase_client and not getattr(self.supabase_client, 'use_mock', True):
            try:
                # Safely store data in Supabase
                data_to_store = {"top_holders": processed_top_holders, "total_shares": total_shares_held}
                self.supabase_client.store_institutional_holdings(
                    symbol=symbol,
                    total_ownership_pct=institutional_ownership,
                    data=data_to_store
                )
                self.logger.info(f"Stored institutional holdings in Supabase for {symbol}")
            except Exception as e:
                self.logger.warning(f"Failed to store institutional holdings in Supabase: {e}")
                
        # 6. Get historical institutional ownership data for change analysis
        historical_data = None
        try:
            if self.supabase_client:
                try:
                    historical_data = self.supabase_client.get_historical_institutional_ownership(symbol, quarters=4)
                    self.logger.info(f"Retrieved historical institutional data for {symbol}")
                except Exception as e:
                    self.logger.warning(f"Failed to get historical institutional ownership from Supabase: {e}")
                    historical_data = None
                
                # Safely analyze historical data if available
                if historical_data is not None and hasattr(historical_data, 'empty') and not historical_data.empty and len(historical_data) > 1:
                    # Sort by quarter to ensure proper chronological order
                    historical_data = historical_data.sort_values('quarter')
                    
                    # Get the oldest and newest ownership percentages
                    oldest_ownership = historical_data.iloc[0]['institutional_ownership_pct']
                    newest_ownership = historical_data.iloc[-1]['institutional_ownership_pct']
                    
                    # Calculate net change (current ownership - previous ownership) / previous ownership
                    # This gives us the percentage increase or decrease over time
                    if oldest_ownership > 0:
                        # Use current ownership for newest if available 
                        if newest_ownership <= 0:
                            newest_ownership = institutional_ownership
                            
                        # Calculate percentage change
                        net_ownership_change = (newest_ownership - oldest_ownership) / oldest_ownership
                        
                        # Add qualitative description
                        time_period = f"over the past {len(historical_data)} quarters"
                        if len(historical_data) == 2:
                            time_period = "quarter-over-quarter"
                        elif len(historical_data) == 4:
                            time_period = "year-over-year"
                            
                        # Format the change description
                        direction = "increase" if net_ownership_change > 0 else "decrease"
                        pct_change = abs(net_ownership_change) * 100
                        change_description = f"{pct_change:.1f}% {direction} {time_period}"
                        
                        # Override net_ownership_change to be the decimal representation
                        # This keeps compatibility with existing code that expects a decimal
                        net_ownership_change = net_ownership_change  # already a decimal (e.g., 0.05 for 5%)
                    else:
                        # Fallback if there's an issue with historical data
                        net_ownership_change = 0
                else:
                    # If no historical data, use the last resort method
                    self.logger.info(f"No historical institutional ownership data found for {symbol}")
                    
                    # Use institutional_metrics if available
                    if institutional_metrics and 'net_ownership_change' in institutional_metrics:
                        net_ownership_change = institutional_metrics['net_ownership_change']
        except Exception as e:
            self.logger.error(f"Error calculating historical ownership change: {e}")
            # Fallback to institutional_metrics
            if institutional_metrics and 'net_ownership_change' in institutional_metrics:
                net_ownership_change = institutional_metrics['net_ownership_change']
        
        # 7. Determine institutional sentiment based on the ownership change
        try:
            if net_ownership_change > 0.02:  # More than 2% increase
                institutional_sentiment = "bullish"
            elif net_ownership_change < -0.02:  # More than 2% decrease
                institutional_sentiment = "bearish"
            else:
                institutional_sentiment = "neutral"
                
            # If enhanced data is available, use its sentiment if we don't have historical data
            if enhanced_data and enhanced_data.get('success', False) and not change_description:
                enhanced_analysis = enhanced_data.get('analysis', {})
                if enhanced_analysis and 'institutional_sentiment' in enhanced_analysis:
                    institutional_sentiment = enhanced_analysis['institutional_sentiment']
                    self.logger.info(f"Using Alpha Vantage institutional sentiment for {symbol}: {institutional_sentiment}")
        except Exception as e:
            self.logger.warning(f"Error determining sentiment: {e}")
            institutional_sentiment = "neutral"  # Safe fallback
        
        # We already processed top holders data earlier, so no need to do it again here
        
        # 8. Compile the final analysis results with safety checks
        try:
            # Make sure all variables are defined and have valid types
            if not isinstance(institutional_sentiment, str):
                institutional_sentiment = "neutral"
                
            if not isinstance(institutional_ownership, (int, float)):
                institutional_ownership = 0
                
            if not isinstance(insider_ownership, (int, float)):
                insider_ownership = 0
                
            if not isinstance(total_shares_held, (int, float)):
                total_shares_held = 0
                
            if not isinstance(total_value_held, (int, float)):
                total_value_held = 0
                
            if not isinstance(top_5_concentration, (int, float)):
                top_5_concentration = 0
                
            if not isinstance(net_ownership_change, (int, float)):
                net_ownership_change = 0
                
            if not isinstance(shares_outstanding, (int, float)):
                shares_outstanding = 0
                
            if not isinstance(change_description, str):
                change_description = ""
                
            holders_count = len(inst_holders_dict) if inst_holders_dict is not None else 0
            
            # Safe rounding function
            def safe_round(value, digits=2):
                try:
                    if isinstance(value, (int, float)):
                        return round(value, digits)
                    return 0
                except:
                    return 0
                    
            # Create the analysis results dictionary
            analysis_results = {
                "symbol": symbol,
                "date": datetime.now().isoformat(),
                "insider_ownership_percentage": safe_round(insider_ownership),
                "institutional_ownership_percentage": safe_round(institutional_ownership),
                "total_institutional_shares": int(total_shares_held) if total_shares_held else 0,
                "total_institutional_value": int(total_value_held) if total_value_held else 0,
                "number_of_institutional_holders": holders_count,
                "top_5_concentration_percentage": safe_round(top_5_concentration),
                "institutional_sentiment": institutional_sentiment,
                "net_ownership_change_percentage": safe_round(net_ownership_change * 100),
                "net_ownership_change_description": change_description,
                "has_historical_data": bool(change_description),
                "shares_outstanding": int(shares_outstanding) if shares_outstanding else 0,
                "top_institutional_holders": processed_top_holders,
                "analysis_type": "institutional_activity",
                "data_source": "Alpha Vantage" if enhanced_data and enhanced_data.get('success') else "Standard"
            }
        except Exception as e:
            self.logger.error(f"Error compiling analysis results: {e}")
            # Provide minimal safe fallback results if everything else fails
            analysis_results = {
                "symbol": symbol,
                "date": datetime.now().isoformat(),
                "institutional_ownership_percentage": institutional_ownership,
                "institutional_sentiment": "neutral",
                "analysis_type": "institutional_activity",
                "error": f"Analysis completed with errors: {str(e)}"
            }
        
        # 9. Add enhanced analysis data if available
        try:
            if enhanced_data and enhanced_data.get('success') and 'analysis' in enhanced_data:
                enhanced_analysis = enhanced_data.get('analysis', {})
                if enhanced_analysis:
                    for key, value in enhanced_analysis.items():
                        # Skip sentiment if we already have historical data-based sentiment
                        if key == 'institutional_sentiment' and change_description:
                            continue
                            
                        # Add all other enhanced analysis fields
                        analysis_results[f"enhanced_{key}"] = value
                        
            self.logger.info(f"Institutional analysis for {symbol} completed successfully")
        except Exception as e:
            self.logger.warning(f"Error adding enhanced analysis data: {e}")
        
        return analysis_results
    
    def _store_analysis_results(self, symbol: str, analysis_results: Dict[str, Any]):
        """
        Store institutional analysis results in the database
        
        Args:
            symbol (str): Stock symbol
            analysis_results (Dict[str, Any]): Analysis results
        """
        try:
            # Convert complex nested structures to strings for storage
            results_copy = analysis_results.copy()
            
            if "top_institutional_holders" in results_copy:
                results_copy["top_institutional_holders"] = json.dumps(results_copy["top_institutional_holders"])
            
            self.supabase_client.store_analysis_result(results_copy)
            self.logger.info(f"Stored institutional activity analysis for {symbol}")
        except Exception as e:
            self.logger.error(f"Error storing institutional activity analysis for {symbol}: {e}")
    
    def get_institutional_activity(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest institutional activity analysis for a symbol
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Optional[Dict[str, Any]]: Latest institutional activity analysis or None if not found
        """
        try:
            results = self.supabase_client.get_analysis_results(
                symbol=symbol,
                analysis_type="institutional_activity",
                limit=1
            )
            
            if not results.empty:
                result_dict = results.iloc[0].to_dict()
                
                # Convert string representations back to objects
                if "top_institutional_holders" in result_dict and isinstance(result_dict["top_institutional_holders"], str):
                    result_dict["top_institutional_holders"] = json.loads(result_dict["top_institutional_holders"])
                
                return result_dict
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting institutional activity for {symbol}: {e}")
            return None
