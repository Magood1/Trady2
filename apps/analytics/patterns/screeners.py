# apps/analytics/patterns/screeners.py
from typing import List

import numpy as np
import pandas as pd

from apps.analytics.patterns.configs import PatternConfigs
from apps.analytics.patterns.types import PatternCandidateData

# ... (دوال find_engulfing و find_doji لم تتغير) ...
def find_engulfing(df: pd.DataFrame) -> List[PatternCandidateData]:
    """
    Vectorized search for bullish and bearish engulfing patterns.

    This function identifies candles where the body completely engulfs the body
    of the previous candle, indicating a potential reversal. The logic is
    fully vectorized for high performance.
    """
    candidates = []
    config = PatternConfigs.get_config()
    body_factor = config.get("ENGULFING_BODY_FACTOR", 1.0)  # Use 1.0 for a strict engulf

    body_size = abs(df["close"] - df["open"])
    prev_body_size = body_size.shift(1)

    # --- Bullish Engulfing Conditions ---
    is_current_green = df["close"] > df["open"]
    is_prev_red = df["close"].shift(1) < df["open"].shift(1)
    # الشرط: جسم الشمعة الحالية يبتلع جسم الشمعة السابقة
    engulfs_body_bullish = (df["close"] > df["open"].shift(1)) & (df["open"] < df["close"].shift(1))
    is_larger_body = body_size > (prev_body_size * body_factor)
    bullish_mask = is_current_green & is_prev_red & engulfs_body_bullish & is_larger_body

    # --- Bearish Engulfing Conditions ---
    is_current_red = df["close"] < df["open"]
    is_prev_green = df["close"].shift(1) > df["open"].shift(1)
    # الشرط: جسم الشمعة الحالية يبتلع جسم الشمعة السابقة
    engulfs_body_bearish = (df["open"] > df["close"].shift(1)) & (df["close"] < df["open"].shift(1))
    # `is_larger_body` هو نفسه
    bearish_mask = is_current_red & is_prev_green & engulfs_body_bearish & is_larger_body

    # --- Process Bullish Matches (Vectorized) ---
    bullish_matches = df[bullish_mask]
    if not bullish_matches.empty:
        bullish_ratios = body_size[bullish_mask] / prev_body_size[bullish_mask]
        bullish_confidence = np.clip((bullish_ratios - 1.0), 0.0, 1.0)

        bullish_candidates = [
            PatternCandidateData(
                symbol="",  # Symbol will be added in the Celery task
                timestamp=ts,
                pattern_type="ENGULFING_BULLISH",
                confidence=float(conf),
                meta={"body_ratio": float(ratio)},
            )
            for ts, conf, ratio in zip(
                bullish_matches.index, bullish_confidence, bullish_ratios
            )
        ]
        candidates.extend(bullish_candidates)

    # --- Process Bearish Matches (Vectorized) ---
    bearish_matches = df[bearish_mask]
    if not bearish_matches.empty:
        bearish_ratios = body_size[bearish_mask] / prev_body_size[bearish_mask]
        bearish_confidence = np.clip((bearish_ratios - 1.0), 0.0, 1.0)

        bearish_candidates = [
            PatternCandidateData(
                symbol="",
                timestamp=ts,
                pattern_type="ENGULFING_BEARISH",
                confidence=float(conf),
                meta={"body_ratio": float(ratio)},
            )
            for ts, conf, ratio in zip(
                bearish_matches.index, bearish_confidence, bearish_ratios
            )
        ]
        candidates.extend(bearish_candidates)

    return sorted(candidates, key=lambda c: c.timestamp)


def find_doji(df: pd.DataFrame) -> List[PatternCandidateData]:
    """
    Vectorized search for Doji candles.

    A Doji is a candle where the open and close are very close, indicating
    indecision in the market. The confidence is higher the smaller the body
    is relative to the total range.
    """
    config = PatternConfigs.get_config()
    threshold = config.get("DOJI_THRESHOLD_RATIO", 0.1)

    body_size = abs(df["close"] - df["open"])
    total_range = df["high"] - df["low"]
    
    # Avoid division by zero for flat candles
    total_range = total_range.replace(0, np.nan)

    doji_mask = (body_size / total_range) < threshold

    # --- Process Doji Matches (Vectorized) ---
    doji_matches = df[doji_mask]
    if doji_matches.empty:
        return []

    body_size_match = body_size[doji_mask]
    total_range_match = total_range[doji_mask]
    
    doji_ratios = body_size_match / total_range_match
    doji_confidence = np.clip(1.0 - (doji_ratios / threshold), 0.0, 1.0)

    candidates = [
        PatternCandidateData(
            symbol="",
            timestamp=ts,
            pattern_type="DOJI",
            confidence=float(conf),
            meta={"body_to_range_ratio": float(ratio)},
        )
        for ts, conf, ratio in zip(doji_matches.index, doji_confidence, doji_ratios)
    ]

    return sorted(candidates, key=lambda c: c.timestamp)

def find_ma_crossover(df: pd.DataFrame) -> List[PatternCandidateData]:
    """
    Vectorized search for Simple Moving Average (SMA) crossovers.
    """
    config = PatternConfigs.get_config()
    fast_window = config.get("MA_CROSS_FAST_WINDOW", 5)
    slow_window = config.get("MA_CROSS_SLOW_WINDOW", 21)

    # --[ تصحيح: زيادة الحد الأدنى للطول لضمان وجود قيمة سابقة صالحة للمقارنة ]--
    if len(df) < slow_window + 1:
        return []

    df_ma = df.copy()
    df_ma["fast_ma"] = df_ma["close"].rolling(window=fast_window).mean()
    df_ma["slow_ma"] = df_ma["close"].rolling(window=slow_window).mean()

    df_ma["prev_fast_ma"] = df_ma["fast_ma"].shift(1)
    df_ma["prev_slow_ma"] = df_ma["slow_ma"].shift(1)

    # --[ تحسين: إزالة القيم الفارغة (NaN) قبل المقارنة لزيادة الموثوقية ]--
    df_ma.dropna(inplace=True)
    if df_ma.empty:
        return []

    bullish_mask = (df_ma["prev_fast_ma"] <= df_ma["prev_slow_ma"]) & \
                   (df_ma["fast_ma"] > df_ma["slow_ma"])

    bearish_mask = (df_ma["prev_fast_ma"] >= df_ma["prev_slow_ma"]) & \
                   (df_ma["fast_ma"] < df_ma["slow_ma"])

    candidates = []
    bullish_matches = df_ma[bullish_mask]
    for ts, row in bullish_matches.iterrows():
        candidates.append(
            PatternCandidateData(
                symbol="",
                timestamp=ts,
                pattern_type="MA_CROSS_BULLISH",
                confidence=1.0,
                meta={"fast_ma": row["fast_ma"], "slow_ma": row["slow_ma"]},
            )
        )

    bearish_matches = df_ma[bearish_mask]
    for ts, row in bearish_matches.iterrows():
        candidates.append(
            PatternCandidateData(
                symbol="",
                timestamp=ts,
                pattern_type="MA_CROSS_BEARISH",
                confidence=1.0,
                meta={"fast_ma": row["fast_ma"], "slow_ma": row["slow_ma"]},
            )
        )

    return sorted(candidates, key=lambda c: c.timestamp)