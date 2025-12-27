# apps/analytics/management/commands/validate_iron_dome.py
import pandas as pd
import numpy as np
from django.core.management.base import BaseCommand
from apps.analytics.services import OHLCVLoader
from apps.market_data.models import Asset
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from datetime import datetime
import pytz

class Command(BaseCommand):
    help = "The Iron Dome Test: Macro Trend + Volatility Ceiling"

    def handle(self, *args, **options):
        symbol = 'XAUUSD'
        
        # Testing the Critical Eras
        eras = {
            "Boring_Era":   ("2018-01-01", "2019-12-31"), # Bear/Range (Should stay quiet)
            "Covid_Shock":  ("2020-01-01", "2021-12-31"), # The Crash & Rebound
            "Geo_Conflict": ("2022-01-01", "2024-12-18"), # The Bull Run
        }

        self.stdout.write(self.style.WARNING(f"--- IRON DOME STRESS TEST: {symbol} ---"))
        self.stdout.write("Filters: 1) Price > Daily EMA200  2) H1 Volatility Not Extreme")
        self.stdout.write("-" * 110)
        self.stdout.write(f"{'Era':<15} | {'Trades':<6} | {'Win%':<6} | {'Net PF':<6} | {'Profit':<10} | {'MaxDD':<8} | {'Status'}")
        self.stdout.write("-" * 110)

        asset = Asset.objects.get(symbol=symbol)
        loader = OHLCVLoader()

        for era_name, (start, end) in eras.items():
            start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
            end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
            
            # 1. Load Data (D1 & H1)
            # Need extra buffer for D1 EMA 200
            load_start = start_dt - pd.Timedelta(days=365)
            
            try:
                df_d1 = loader.load_dataframe(asset, 'D1', load_start, end_dt)
                df_h1 = loader.load_dataframe(asset, 'H1', start_dt, end_dt)
            except Exception:
                self.stdout.write(f"{era_name:<15} | NO DATA")
                continue
                
            if len(df_d1) < 200 or len(df_h1) < 500:
                self.stdout.write(f"{era_name:<15} | INSUFFICIENT DATA")
                continue

            # --- 2. MACRO FILTER (THE IRON DOME) ---
            # Calculate D1 EMA 200
            d1_ema200 = EMAIndicator(close=df_d1['close'], window=200).ema_indicator()
            
            # Filter: Bull Market Only
            # Shift by 1 to avoid lookahead (Trade today based on yesterday's close)
            df_d1['bull_regime'] = (df_d1['close'] > d1_ema200).shift(1).fillna(0)
            
            # Align to H1
            regime_aligned = df_d1['bull_regime'].reindex(df_h1.index, method='ffill').fillna(0)
            df_h1['macro_bull'] = regime_aligned

            # --- 3. MICRO FILTERS (H1) ---
            close = df_h1['close']
            
            # Indicators
            h1_ema200 = EMAIndicator(close=close, window=200).ema_indicator()
            h1_rsi = RSIIndicator(close=close, window=14).rsi()
            h1_adx = ADXIndicator(high=df_h1['high'], low=df_h1['low'], close=close, window=14).adx()
            
            # Volatility Ceiling (Avoid Panic Days)
            # ATR 14 on H1
            h1_atr = AverageTrueRange(high=df_h1['high'], low=df_h1['low'], close=close, window=14).average_true_range()
            # Dynamic Cap: 2x the rolling average of ATR (Relative Volatility)
            atr_ma = h1_atr.rolling(window=100).mean()
            vol_safe = h1_atr < (atr_ma * 2.0)

            # --- 4. SIGNALS ---
            # Signal:
            # 1. MACRO BULL (D1 > EMA200)
            # 2. MICRO TREND (H1 > EMA200)
            # 3. PULLBACK (RSI < 45)
            # 4. STRENGTH (ADX > 20)
            # 5. SAFE CONDITIONS (Vol Ceiling)
            # 6. CONFIRMATION (Green Candle)
            
            signals = (df_h1['macro_bull'] == 1) & \
                      (close > h1_ema200) & \
                      (h1_rsi < 45) & \
                      (h1_adx > 20) & \
                      (vol_safe) & \
                      (close > df_h1['open'])
            
            # --- 5. SIMULATION ---
            trades = []
            equity = 10000.0
            COSTS = 0.50 
            
            indices = np.where(signals)[0]
            
            # Volatility for stops (Log Returns Std)
            vol_series = close.pct_change().rolling(20).std()
            
            for idx in indices:
                if idx + 24 >= len(df_h1): continue
                
                entry_price = float(df_h1['open'].iloc[idx+1])
                vol = float(vol_series.iloc[idx])
                if np.isnan(vol) or vol == 0: continue
                
                # Dynamic Risk
                stop_dist = max(2.0, entry_price * vol * 2.0)
                sl = entry_price - stop_dist
                tp = entry_price + (stop_dist * 2.0) # 2R
                
                outcome_pnl = 0
                
                for i in range(idx+1, idx+25):
                    row = df_h1.iloc[i]
                    if row['low'] <= sl:
                        outcome_pnl = (sl - entry_price) - COSTS
                        break
                    elif row['high'] >= tp:
                        outcome_pnl = (tp - entry_price) - COSTS
                        break
                
                if outcome_pnl == 0:
                    outcome_pnl = (float(close.iloc[idx+24]) - entry_price) - COSTS
                
                risk_amt = equity * 0.005 # 0.5% Risk
                units = risk_amt / stop_dist
                pnl = outcome_pnl * units
                
                trades.append(pnl)
                equity += pnl

            # --- 6. REPORTING ---
            if not trades:
                status = "PASS (SAFE)" if era_name in ["Boring_Era", "Covid_Shock"] else "NO TRADES"
                self.stdout.write(f"{era_name:<15} | 0      | N/A    | N/A    | $0         | 0.0%     | {status}")
                continue
                
            wins = sum(1 for t in trades if t > 0)
            losses = sum(1 for t in trades if t <= 0)
            gross_win = sum(t for t in trades if t > 0)
            gross_loss = abs(sum(t for t in trades if t <= 0))
            
            pf = gross_win / gross_loss if gross_loss > 0 else 99.9
            net = sum(trades)
            
            # Max DD
            eq_curve = pd.Series([10000] + trades).cumsum()
            dd = (eq_curve - eq_curve.cummax()) / eq_curve.cummax()
            max_dd = dd.min()
            
            status = "FAIL"
            if pf >= 1.3 and max_dd > -0.10: status = "PASS"
            elif pf >= 1.1: status = "WEAK"
            
            # Override for Boring/Bear Eras: Survival is Key
            if era_name == "Boring_Era" and max_dd > -0.05:
                status = "PASS (SURVIVED)"
            
            color = self.style.SUCCESS if "PASS" in status else self.style.ERROR
            self.stdout.write(color(f"{era_name:<15} | {len(trades):<6} | {wins/len(trades):.1%}  | {pf:.2f}   | ${net:<9.0f} | {max_dd:.1%}   | {status}"))

            