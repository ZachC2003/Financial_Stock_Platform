"""
Deterministic calculation utilities

This module provides consistent calculation methods that produce the same result
every time they're called with the same inputs, eliminating randomness.

These utilities ensure that financial metrics in the platform remain consistent
between refreshes but can still evolve over time in a realistic manner.
"""

def calculate_deterministic_relative_strength(ticker: str) -> float:
    """
    Calculate a deterministic relative strength value based solely on ticker symbol.
    This ensures consistent values between refreshes.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        float: Relative strength value (typically between 0.8 and 1.2)
    """
    # Use ticker characters to create a consistent hash value
    ticker_chars = sum(ord(c) for c in ticker.upper())
    
    # Normalize to common relative strength range (0.8 to 1.2)
    base_value = 0.8 + ((ticker_chars % 40) / 100)  # Range: 0.8 to 1.19
    
    # Apply a modifier based on ticker length
    length_modifier = min(0.1, len(ticker) / 100)  # Small modifier based on length
    
    # Special handling for well-known stocks
    well_performing = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    if ticker.upper() in well_performing:
        # Slight boost for well-known outperformers
        return round(min(1.2, base_value + length_modifier + 0.05), 2)
    
    return round(min(1.2, base_value + length_modifier), 2)

def calculate_deterministic_net_ownership_change(ticker: str) -> float:
    """
    Calculate a deterministic net ownership change based on ticker symbol and current time period.
    This ensures consistent values for the same time period, but allows for change over time.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        float: Net ownership change percentage (typically between -0.03 and +0.05)
    """
    from datetime import datetime
    
    # Use ticker characters to create a consistent hash value
    ticker_chars = sum(ord(c) for c in ticker.upper())
    
    # Get current year and month to create time periods
    current_year = datetime.now().year
    current_month = datetime.now().month
    current_week = (datetime.now().day - 1) // 7 + 1  # Week of the month (1-5)
    
    # Create a time factor that changes monthly
    # Use a deterministic approach where each month's value is derived from the combination of
    # ticker characteristics and the current month
    month_seed = (ticker_chars + current_month + current_year) % 12
    monthly_factor = (month_seed - 6) / 200  # Range: -0.03 to +0.03 with 0.005 steps
    
    # Add a smaller weekly variation
    week_seed = (ticker_chars + current_week + current_month) % 5
    weekly_factor = (week_seed - 2) / 400  # Range: -0.005 to +0.005 with 0.0025 steps
    
    # Base value between -0.02 and +0.04 based on ticker
    base_value = ((ticker_chars % 6) - 2) / 100
    
    # Small fixed adjustment based on ticker length
    length_adjustment = len(ticker) / 1000
    
    # Combine all factors for the final value
    # This will be consistent for the same ticker within the same week
    # but change predictably from week to week and month to month
    net_change = base_value + length_adjustment + monthly_factor + weekly_factor
    
    # Ensure realistic bounds and round to 3 decimal places
    return round(max(-0.04, min(0.06, net_change)), 3)


def calculate_deterministic_beta(ticker: str) -> float:
    """
    Calculate a deterministic beta value based on ticker symbol and characteristics.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        float: Beta value (typically between 0.5 and 2.0)
    """
    # Create a hash value based on ticker
    ticker_chars = sum(ord(c) for c in ticker.upper())
    
    # Different stock sectors tend to have different beta ranges
    # Tech stocks typically have higher betas, utilities lower
    tech_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD']
    financial_tickers = ['JPM', 'BAC', 'GS', 'MS', 'WFC', 'C']
    utility_tickers = ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC']
    healthcare_tickers = ['JNJ', 'PFE', 'MRK', 'ABBV', 'LLY', 'BMY']
    
    # Set base range based on sector
    if ticker.upper() in tech_tickers:
        base_min, base_max = 1.0, 2.0  # Tech stocks - higher beta
    elif ticker.upper() in financial_tickers:
        base_min, base_max = 0.9, 1.5  # Financial - medium-high beta
    elif ticker.upper() in utility_tickers:
        base_min, base_max = 0.5, 0.8  # Utilities - low beta
    elif ticker.upper() in healthcare_tickers:
        base_min, base_max = 0.7, 1.1  # Healthcare - medium-low beta
    else:
        base_min, base_max = 0.8, 1.3  # Default range
    
    # Calculate a specific value within the range based on ticker hash
    hash_normalized = (ticker_chars % 100) / 100.0  # 0.0 to 0.99
    beta = base_min + hash_normalized * (base_max - base_min)
    
    return round(beta, 2)


def calculate_deterministic_earnings_surprise(ticker: str, quarter: int = None) -> float:
    """
    Calculate a deterministic earnings surprise percentage.
    
    Args:
        ticker (str): Stock ticker symbol
        quarter (int, optional): Calendar quarter (1-4). If None, uses current quarter.
        
    Returns:
        float: Earnings surprise percentage (-20 to +20)
    """
    from datetime import datetime
    
    # Use current quarter if not specified
    if quarter is None:
        current_month = datetime.now().month
        quarter = ((current_month - 1) // 3) + 1  # 1-4
    
    # Get consistent values based on ticker and quarter
    ticker_chars = sum(ord(c) for c in ticker.upper())
    quarter_seed = (ticker_chars + quarter) % 100
    
    # Companies with stronger fundamentals tend to beat more often
    strong_companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JNJ', 'V']
    
    # Different calculation based on company strength
    if ticker.upper() in strong_companies:
        # Strong companies beat more often with higher percentages
        beat_chance = 75  # 75% chance of beating
        if quarter_seed < beat_chance:
            # Beat expectations (positive surprise)
            surprise_pct = 2.0 + (quarter_seed % 12)  # Range: +2% to +14%
        else:
            # Miss expectations (negative surprise)
            surprise_pct = -1.0 - (quarter_seed % 7)   # Range: -1% to -8%
    else:
        # Average companies have more variability
        beat_chance = 60  # 60% chance of beating
        if quarter_seed < beat_chance:
            # Beat expectations (positive surprise)
            surprise_pct = 1.0 + (quarter_seed % 10)  # Range: +1% to +11%
        else:
            # Miss expectations (negative surprise)
            surprise_pct = -1.0 - (quarter_seed % 10)  # Range: -1% to -11%
    
    return round(surprise_pct, 1)


def calculate_deterministic_rsi(ticker: str, time_period: str = "weekly") -> float:
    """
    Calculate a deterministic RSI (Relative Strength Index) value.
    
    Args:
        ticker (str): Stock ticker symbol
        time_period (str): Time period for the calculation ('daily', 'weekly', 'monthly')
        
    Returns:
        float: RSI value (0-100)
    """
    from datetime import datetime
    
    # Create base values from ticker
    ticker_chars = sum(ord(c) for c in ticker.upper())
    
    # Add time-based component
    current_day = datetime.now().day
    current_month = datetime.now().month
    current_year = datetime.now().year
    
    if time_period == "daily":
        time_factor = current_day + current_month * 31 + current_year * 365
    elif time_period == "weekly":
        week_number = (current_day - 1) // 7 + 1
        time_factor = week_number + current_month * 5 + current_year * 60
    else:  # monthly
        time_factor = current_month + current_year * 12
    
    # Combine ticker characteristics with time factor
    combined_seed = (ticker_chars + time_factor) % 100
    
    # Calculate mid-range RSI with oscillation
    # Use sine wave pattern for more realistic values that oscillate over time
    import math
    oscillation = math.sin(combined_seed / 15.9) * 20  # -20 to +20 oscillation
    
    # Base RSI value depends on ticker characteristics
    base_rsi = 50 + (ticker_chars % 20 - 10)  # 40-60 range for base
    
    # Calculate final RSI with limits
    rsi = base_rsi + oscillation
    rsi = max(0, min(100, rsi))  # RSI must be between 0-100
    
    return round(rsi, 1)
