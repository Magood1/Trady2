# apps/trading_core/strategies.py
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator

def calculate_choppiness(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Manual CHOP calculation for Strategy use."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    tr_sum = tr.rolling(window=window).sum()
    range_hl = high.rolling(window=window).max() - low.rolling(window=window).min()
    
    # Handle zeros to avoid log errors
    range_hl = range_hl.replace(0, np.nan)
    
    chop = 100 * np.log10(tr_sum / range_hl) / np.log10(window)
    return chop.fillna(50) # Default to neutral if NaN

def trend_pullback_signal(df: pd.DataFrame) -> bool:
    """
    Trend Pullback + Quality Control Strategy (Robust).
    """
    if len(df) < 200: 
        return False

    close = df['close']
    open_price = df['open']
    high = df['high']
    low = df['low']
    
    # Indicators
    ema_200 = EMAIndicator(close=close, window=200).ema_indicator()
    rsi_ind = RSIIndicator(close=close, window=14)
    adx_ind = ADXIndicator(high=high, low=low, close=close, window=14)
    
    # Calculate Chop Manually
    chop_series = calculate_choppiness(df)
    
    # Latest Values
    last_close = close.iloc[-1]
    last_open = open_price.iloc[-1]
    last_ema = ema_200.iloc[-1]
    last_rsi = rsi_ind.rsi().iloc[-1]
    last_adx = adx_ind.adx().iloc[-1]
    last_chop = chop_series.iloc[-1]
    
    # Logic Checks
    is_uptrend = last_close > last_ema
    is_pullback = last_rsi < 45
    is_strong_regime = last_adx > 20
    is_linear_trend = last_chop < 50 # THE GUARD
    is_green_candle = last_close > last_open 
    
    if is_uptrend and is_pullback and is_strong_regime and is_linear_trend and is_green_candle:
        return True
        
    return False

# Legacy
def bollinger_breakout_signal(df: pd.DataFrame, window: int = 20, dev: int = 2) -> bool:
    return False

