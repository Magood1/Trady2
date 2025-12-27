# apps/analytics/quant/pairs_trading.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

class PairsTradingEngine:
    """
    Rolling OLS Engine for Statistical Arbitrage.
    Calculates dynamic Hedge Ratio and Z-Score.
    """
    
    @staticmethod
    def get_aligned_data(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """
        Aligns two dataframes on timestamp (Inner Join).
        """
        d1 = df1[['close']].rename(columns={'close': 'y'}) # Target (e.g. Gold)
        d2 = df2[['close']].rename(columns={'close': 'x'}) # Hedge (e.g. Silver)
        
        # Inner join ensures valid data for both
        df = pd.concat([d1, d2], axis=1, join='inner').dropna()
        return df

    @staticmethod
    def calculate_rolling_metrics(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """
        Calculates Rolling Alpha, Beta, Spread, and Z-Score.
        """
        # FIX: Explicit casting to float to ensure NumPy compatibility
        # If 'y' or 'x' are Decimal type, np.log will fail.
        y = np.log(df['y'].astype(float))
        x = np.log(df['x'].astype(float))
        
        # 1. Rolling Statistics
        cov = x.rolling(window=window).cov(y)
        var = x.rolling(window=window).var()
        mean_x = x.rolling(window=window).mean()
        mean_y = y.rolling(window=window).mean()
        
        # 2. Rolling Beta (Hedge Ratio) = Cov(x,y) / Var(x)
        beta = cov / var
        
        # 3. Rolling Alpha
        alpha = mean_y - (beta * mean_x)
        
        # 4. Construct the Spread
        # Spread = Y - (Alpha + Beta * X)
        spread = y - (alpha + beta * x)
        
        # 5. Z-Score
        spread_mean = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std()
        
        z_score = (spread - spread_mean) / spread_std
        
        result = df.copy()
        result['beta'] = beta
        result['spread'] = spread
        result['z_score'] = z_score
        
        return result.dropna()

    @staticmethod
    def check_cointegration(series: pd.Series) -> float:
        try:
            if len(series) > 5000:
                s = series.iloc[::5]
            else:
                s = series.astype(float) # Ensure float
            
            result = adfuller(s)
            return result[1] 
        except Exception:
            return 1.0
        