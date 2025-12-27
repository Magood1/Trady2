# apps/analytics/features/pipeline.py
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator
# REMOVED: from ta.trend import ChoppinessIndicator
from apps.market_data.models import Asset

class FeaturePipeline:
    @staticmethod
    def _calculate_choppiness(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Manual implementation of Choppiness Index (CHOP).
        Formula: 100 * LOG10(Sum(TR, n) / (Max(H, n) - Min(L, n))) / LOG10(n)
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # 1. True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 2. Sum of TR
        tr_sum = tr.rolling(window=window).sum()
        
        # 3. Range (Max High - Min Low)
        max_high = high.rolling(window=window).max()
        min_low = low.rolling(window=window).min()
        range_hl = max_high - min_low
        
        # Avoid division by zero
        range_hl = range_hl.replace(0, np.nan)
        
        # 4. Calculation
        chop = 100 * np.log10(tr_sum / range_hl) / np.log10(window)
        return chop

    @staticmethod
    def build_feature_dataframe(symbol: str, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        if ohlcv_df.empty:
            return pd.DataFrame()

        df = ohlcv_df.copy()
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 1. Volatility
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['vol_std'] = df['log_ret'].rolling(window=20).std()
        
        # 2. Trend Context
        ema_200 = EMAIndicator(close=df['close'], window=200).ema_indicator()
        df['dist_ema200'] = (df['close'] - ema_200) / ema_200
        
        # 3. Momentum
        rsi_ind = RSIIndicator(close=df['close'], window=14)
        df['rsi'] = rsi_ind.rsi()
        
        # 4. Trend Strength
        adx_ind = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['adx'] = adx_ind.adx()
        
        # 5. Quality (Manual Choppiness)
        df['chop'] = FeaturePipeline._calculate_choppiness(df)

        # 6. Interaction
        df['is_green'] = (df['close'] > df['open']).astype(float)

        # Cleaning
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        
        final_cols = ['vol_std', 'dist_ema200', 'rsi', 'adx', 'chop', 'is_green']
        
        for col in final_cols:
            if col not in df.columns:
                df[col] = 0.0
                
        return df[final_cols]
    