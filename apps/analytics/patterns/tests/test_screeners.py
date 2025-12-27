import datetime

import pandas as pd
import pytest
from django.utils import timezone

from apps.analytics.patterns.screeners import find_doji, find_engulfing, find_ma_crossover

@pytest.fixture
def sample_df() -> pd.DataFrame:
    # ... (الكود لم يتغير)
    data = [
        {"open": 102, "high": 103, "low": 100, "close": 101},
        {"open": 100, "high": 105, "low": 99, "close": 104},
        {"open": 105, "high": 108, "low": 102, "close": 105.1},
        {"open": 110, "high": 115, "low": 109, "close": 114},
        {"open": 115, "high": 116, "low": 108, "close": 109},
    ]
    timestamps = [timezone.now() - datetime.timedelta(hours=i) for i in range(len(data), 0, -1)]
    return pd.DataFrame(data, index=pd.Index(timestamps, name="timestamp"))

def test_find_engulfing(sample_df):
    # ... (الكود لم يتغير)
    candidates = find_engulfing(sample_df)
    assert len(candidates) == 2
    assert candidates[0].pattern_type == "ENGULFING_BULLISH"
    assert candidates[1].pattern_type == "ENGULFING_BEARISH"

def test_find_doji(sample_df):
    # ... (الكود لم يتغير)
    candidates = find_doji(sample_df)
    assert len(candidates) == 1
    assert candidates[0].pattern_type == "DOJI"

# --[ تصحيح: استخدام بيانات اختبار أطول وأكثر قوة ]--
def test_find_ma_crossover_bullish():
    """Tests only the bullish (golden) crossover event with robust data."""
    # 25 نقطة بيانات لتوفير فترة استقرار كافية للمتوسط البطيء (21)
    close_prices = ([100] * 23) + [105, 110] 
    timestamps = [timezone.now() - datetime.timedelta(hours=i) for i in range(len(close_prices), 0, -1)]
    df = pd.DataFrame({
        "open": 1, "high": 1, "low": 1, "close": close_prices
    }, index=pd.Index(timestamps, name="timestamp"))
    
    candidates = find_ma_crossover(df)
    
    assert len(candidates) == 1
    assert candidates[0].pattern_type == "MA_CROSS_BULLISH"
    assert candidates[0].timestamp == df.index[-2] # التقاطع يحدث عند 105

def test_find_ma_crossover_bearish():
    """Tests only the bearish (death) crossover event with robust data."""
    close_prices = ([100] * 23) + [95, 90]
    timestamps = [timezone.now() - datetime.timedelta(hours=i) for i in range(len(close_prices), 0, -1)]
    df = pd.DataFrame({
        "open": 1, "high": 1, "low": 1, "close": close_prices
    }, index=pd.Index(timestamps, name="timestamp"))

    candidates = find_ma_crossover(df)
    
    assert len(candidates) == 1
    assert candidates[0].pattern_type == "MA_CROSS_BEARISH"
    assert candidates[0].timestamp == df.index[-2] # التقاطع يحدث عند 95