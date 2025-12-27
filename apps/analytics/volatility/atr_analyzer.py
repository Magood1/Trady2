#apps/analytics/volatility/atrs.py
import numpy as np
import pandas as pd


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """
    Calculate the Average True Range (ATR).

    Args:
        high: A pandas Series of high prices.
        low: A pandas Series of low prices.
        close: A pandas Series of close prices.
        window: The lookback period for the ATR calculation.

    Returns:
        A pandas Series containing the ATR values.
    """
    if not all(isinstance(s, pd.Series) for s in [high, low, close]):
        raise TypeError("Inputs must be pandas Series.")

    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()
