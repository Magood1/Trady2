# apps/analytics/services.py
"""
Analytics services: OHLCV loading, feature engineering, regime classification,
and chart data assembly. This version includes a composite 'overbought' feature.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from django.conf import settings
from django.db import transaction
from django.utils import timezone

# استيراد مكتبات المؤشرات الفنية
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands

from apps.analytics.serializers import AnnotationSerializer, OHLCVChartSerializer
from apps.analytics.volatility.atr_analyzer import atr as calculate_atr
from apps.market_data.models import Asset, MarketRegime, OHLCV
from apps.trading_core.models import PatternCandidate, VerifiedPattern, TradingSignal

logger = structlog.get_logger(__name__)


# ---------------------------
# OHLCV Loader
# ---------------------------
class OHLCVLoader:
    """
    Loads OHLCV rows for a given asset/timeframe and returns a timezone-aware (UTC) DataFrame.
    """

    def load_dataframe(
        self,
        asset: Asset,
        timeframe: str,
        start_utc: datetime.datetime,
        end_utc: datetime.datetime,
    ) -> pd.DataFrame:
        logger.info("Loading OHLCV data for analysis", asset=asset.symbol, timeframe=timeframe)

        queryset = OHLCV.objects.filter(
            asset=asset, timeframe=timeframe, timestamp__range=(start_utc, end_utc)
        ).order_by("timestamp")

        if not queryset.exists():
            raise ValueError(f"No OHLCV data found for {asset.symbol} in the given range.")

        df = pd.DataFrame.from_records(queryset.values("timestamp", "open", "high", "low", "close", "volume"))

        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df.set_index("timestamp", inplace=True)

        df = df[["open", "high", "low", "close", "volume"]]
        return df


# ---------------------------
# Feature Engineering (Sprint 5.4 - Signal Quality Refinement)
# ---------------------------
class FeatureEngineer:
    """
    Adds a comprehensive set of features, now including a composite
    'overbought' indicator to improve signal quality.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or settings.ANALYTICS_CONFIG

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df_featured = df.copy()

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df_featured.columns:
                df_featured[col] = pd.to_numeric(df_featured[col], errors='coerce')

        # === 1. Standard Technical Indicators ===
        df_featured["atr"] = calculate_atr(df_featured["high"], df_featured["low"], df_featured["close"], window=self.config.get("ATR_WINDOW", 14))
        df_featured["rsi"] = RSIIndicator(close=df_featured["close"], window=self.config.get("RSI_WINDOW", 14)).rsi()
        
        macd = MACD(close=df_featured["close"], window_slow=26, window_fast=12, window_sign=9)
        df_featured["macd_diff"] = macd.macd_diff()
        
        stoch = StochasticOscillator(high=df_featured["high"], low=df_featured["low"], close=df_featured["close"], window=14, smooth_window=3)
        df_featured["stoch_k"] = stoch.stoch()
        df_featured["stoch_d"] = stoch.stoch_signal()

        bb = BollingerBands(close=df_featured["close"], window=20, window_dev=2)
        df_featured["bb_width"] = bb.bollinger_wband()
        df_featured["bb_pband"] = bb.bollinger_pband()

        # === 2. Advanced Volatility Features ===
        log_high = np.log(df_featured['high'].clip(lower=1e-9))
        log_low = np.log(df_featured['low'].clip(lower=1e-9))
        log_close = np.log(df_featured['close'].clip(lower=1e-9))
        log_open = np.log(df_featured['open'].clip(lower=1e-9))
        df_featured['garman_klass_vol'] = ((log_high - log_low)**2) / 2 - (2*np.log(2) - 1) * ((log_close - log_open)**2)

        # === 3. Multi-Horizon Momentum Features ===
        df_featured['log_ret_1h'] = np.log(df_featured['close'].clip(lower=1e-9) / df_featured['close'].shift(1).clip(lower=1e-9))
        for lag in [3, 6, 12, 24, 24*5, 24*21]:
            if len(df_featured) > lag:
                df_featured[f'log_ret_{lag}h'] = np.log(df_featured['close'].clip(lower=1e-9) / df_featured['close'].shift(lag).clip(lower=1e-9))
                df_featured[f'momentum_{lag}h'] = df_featured['log_ret_1h'].rolling(window=lag).mean()

        # === 4. Mean-Reversion / Value Features ===
        for window in [20, 50, 200]:
            sma = df_featured['close'].rolling(window=window).mean()
            df_featured[f'mean_reversion_dist_{window}'] = (df_featured['close'] - sma) / sma
            if window == 200:
                df_featured['momentum_filter_200'] = (df_featured['close'] > sma).astype(float)

        # === 5. **الميزة المركبة الجديدة (توصية الخبراء)** ===
        # إنشاء ميزة "ضوء أحمر" للشراء في ظروف خطرة
        # الشروط: السعر فوق 95% من نطاق بولينجر، ومؤشر القوة النسبية فوق 65
        overbought_condition = (df_featured['bb_pband'] > 0.95) & (df_featured['rsi'] > 65)
        df_featured['extreme_overbought_signal'] = overbought_condition.astype(float)
        
        # === 6. Time-Based Features ===
        df_featured['hour_of_day'] = df_featured.index.hour
        df_featured['day_of_week'] = df_featured.index.dayofweek
        
        # === 7. Final Data Sanitization ===
        df_featured = df_featured.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
            
        return df_featured


# ---------------------------
# Legacy Placeholders
# ---------------------------
class RegimeClassifier:
    pass

def run_regime_analysis_for_asset(asset: Asset, timeframe: str) -> None:
    pass


# ---------------------------
# Chart Data Service
# ---------------------------
class ChartDataService:
    @staticmethod
    def get_chart_data(symbol: str, timeframe: str, start_utc: datetime.datetime, end_utc: datetime.datetime) -> Dict[str, Any]:
        asset = Asset.objects.get(symbol__iexact=symbol)
        ohlcv_qs = OHLCV.objects.filter(
            asset=asset, timeframe=timeframe, timestamp__range=(start_utc, end_utc)
        ).order_by("timestamp")

        if not ohlcv_qs.exists():
            raise ValueError("No OHLCV data for the selected range.")

        ohlcv_data = OHLCVChartSerializer(ohlcv_qs, many=True).data
        annotations = []
        annotations.extend(ChartDataService._get_regime_annotations(asset, start_utc, end_utc))
        annotations.extend(ChartDataService._get_verified_pattern_annotations(asset, start_utc, end_utc))
        annotations.extend(ChartDataService._get_signal_annotations(asset, start_utc, end_utc))
        serialized_annotations = AnnotationSerializer(sorted(annotations, key=lambda x: x["time"]), many=True).data

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "ohlcv": ohlcv_data,
            "annotations": serialized_annotations,
        }

    @staticmethod
    def _get_regime_annotations(asset: Asset, start: datetime.datetime, end: datetime.datetime) -> List[Dict[str, Any]]:
        regimes = MarketRegime.objects.filter(asset=asset, timestamp__range=(start, end))
        regime_map = {"TRENDING": "#26A69A", "MEAN_REVERTING": "#FFAB00", "HIGH_VOLATILITY": "#EF5350", "RANDOM": "#BDBDBD"}
        return [{"time": r.timestamp, "text": f"Regime: {r.regime}", "color": regime_map.get(r.regime, "grey"), "position": "belowBar", "shape": "diamond"} for r in regimes]

    @staticmethod
    def _get_verified_pattern_annotations(asset: Asset, start: datetime.datetime, end: datetime.datetime) -> List[Dict[str, Any]]:
        verified_qs = VerifiedPattern.objects.filter(candidate__asset=asset, candidate__timestamp__range=(start, end)).select_related("candidate")
        return [{"time": vp.candidate.timestamp, "text": f"Verified: {vp.candidate.pattern_type}<br>Conf: {vp.confidence:.2f}", "color": "#9C27B0", "position": "aboveBar", "shape": "arrowDown"} for vp in verified_qs]

    @staticmethod
    def _get_signal_annotations(asset: Asset, start: datetime.datetime, end: datetime.datetime) -> List[Dict[str, Any]]:
        signals = TradingSignal.objects.filter(asset=asset, timestamp__range=(start, end))
        annotations = []
        for s in signals:
            shape = "arrowUp" if s.signal_type == "BUY" else "arrowDown"
            color = "rgba(46, 204, 113, 0.9)" if s.signal_type == "BUY" else "rgba(231, 76, 60, 0.9)"
            position = "belowBar" if s.signal_type == "BUY" else "aboveBar"
            text = f"<b>SIGNAL: {s.signal_type}</b><br>Entry: {s.entry_price:.5f}<br>SL: {s.stop_loss:.5f} | TP: {s.take_profit:.5f}"
            annotations.append({"time": s.timestamp, "text": text, "color": color, "position": position, "shape": shape})
        return annotations
    
    