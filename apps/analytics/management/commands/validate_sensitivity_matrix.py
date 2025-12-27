# apps/analytics/management/commands/validate_sensitivity_matrix.py
import pandas as pd
import numpy as np
from django.core.management.base import BaseCommand
from apps.analytics.services import OHLCVLoader
from apps.market_data.models import Asset
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from datetime import datetime, time
import pytz
from itertools import product

class Command(BaseCommand):
    help = "Governance Protocol: Sensitivity Matrix & Stress Test"

    def handle(self, *args, **options):
        symbol = 'XAUUSD'
        
        # 1. THE STRESS PARAMETERS (Cost Pressure)
        # Spread +25%, Slippage x1.5
        BASE_SPREAD = 0.30 * 1.25  # $0.375
        BASE_COMM = 0.10           # $0.10
        SLIP_FACTOR = 0.1 * 1.5    # 15% of Volatility
        TOTAL_FIXED_COST = BASE_SPREAD + BASE_COMM

        # 2. SENSITIVITY GRID
        ema_windows = [50, 100, 150]      # D1 Trend Filter
        atr_caps = [1.8, 2.0, 2.2]        # H1 Volatility Ceiling multipliers
        
        # Eras to test (Focus on the problematic ones + success one)
        eras = {
            "Covid_Shock":  ("2020-01-01", "2021-12-31"), # The Stress Test
            "Geo_Conflict": ("2022-01-01", "2024-12-18"), # The Benchmark
        }

        self.stdout.write(self.style.ERROR(f"--- SENSITIVITY MATRIX: {symbol} ---"))
        self.stdout.write(f"Cost Basis: Spread=${BASE_SPREAD:.3f} | SlipFactor={SLIP_FACTOR}")
        self.stdout.write(f"Filters: Session(8-20 UTC) + NewsBlock(4h after 3xATR spike)")
        self.stdout.write("-" * 100)
        self.stdout.write(f"{'Era':<12} | {'EMA':<3} | {'ATR':<3} | {'Trades':<6} | {'Win%':<6} | {'PF':<6} | {'MaxDD':<8} | {'Status'}")
        self.stdout.write("-" * 100)

        asset = Asset.objects.get(symbol=symbol)
        loader = OHLCVLoader()

        # Load Full Data once to optimize
        full_start = datetime(2019, 1, 1, tzinfo=pytz.UTC)
        full_end = datetime.now(pytz.UTC)
        
        try:
            # Need D1 for EMA, H1 for execution
            raw_d1 = loader.load_dataframe(asset, 'D1', full_start, full_end)
            raw_h1 = loader.load_dataframe(asset, 'H1', full_start, full_end)
        except Exception:
            self.stdout.write("CRITICAL: Missing Data")
            return

        # Pre-calculate H1 fixed indicators
        h1_rsi = RSIIndicator(raw_h1['close'], window=14).rsi()
        h1_adx = ADXIndicator(raw_h1['high'], raw_h1['low'], raw_h1['close'], window=14).adx()
        h1_atr = AverageTrueRange(raw_h1['high'], raw_h1['low'], raw_h1['close'], window=14).average_true_range()
        h1_ema200 = EMAIndicator(raw_h1['close'], window=200).ema_indicator()
        
        # Volatility Spike Detector (News Proxy)
        # If High-Low > 3 * ATR, block next 4 hours
        bar_range = raw_h1['high'] - raw_h1['low']
        is_shock = bar_range > (h1_atr * 3.0)
        # Create blocking mask (forward fill 4 bars)
        shock_block = is_shock.rolling(window=4, closed='left').max().fillna(0) # 1 if shock in last 4 bars

        # --- RUN THE MATRIX ---
        for era_name, (start_str, end_str) in eras.items():
            start_dt = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
            end_dt = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
            
            # Slice Era
            mask_h1 = (raw_h1.index >= start_dt) & (raw_h1.index <= end_dt)
            if mask_h1.sum() < 500: continue
            
            era_h1 = raw_h1[mask_h1]
            
            for ema_win, atr_mult in product(ema_windows, atr_caps):
                
                # 1. D1 Macro Filter (Dynamic)
                d1_ema = EMAIndicator(raw_d1['close'], window=ema_win).ema_indicator()
                # Shift 1 day
                d1_bull = (raw_d1['close'] > d1_ema).shift(1)
                # Align
                macro_filter = d1_bull.reindex(era_h1.index, method='ffill').fillna(0)
                
                # 2. H1 Volatility Cap (Dynamic)
                atr_ceiling = h1_atr.rolling(window=100).mean() * atr_mult
                vol_safe = h1_atr[mask_h1] < atr_ceiling[mask_h1]
                
                # 3. Session Filter (Liquidity Gate)
                # Allow only 08:00 to 20:00 UTC (London + NY)
                # This avoids Asian session spreads and low liquidity chops
                hour_filter = pd.Series(era_h1.index.hour, index=era_h1.index).between(8, 20)
                
                # 4. News Block
                news_safe = shock_block[mask_h1] == 0
                
                # --- COMBINE SIGNALS ---
                # Logic: Macro Bull + H1 Uptrend + Pullback + Strength + Green Candle + ALL GATES
                signals = (macro_filter == 1) & \
                          (era_h1['close'] > h1_ema200[mask_h1]) & \
                          (h1_rsi[mask_h1] < 45) & \
                          (h1_adx[mask_h1] > 20) & \
                          (era_h1['close'] > era_h1['open']) & \
                          (vol_safe) & \
                          (hour_filter) & \
                          (news_safe)
                
                # --- SIMULATION ---
                trades = []
                equity = 10000.0
                indices = np.where(signals)[0]
                
                # H1 Vol for execution
                vol_series = era_h1['close'].pct_change().rolling(20).std()
                
                for idx in indices:
                    if idx + 24 >= len(era_h1): continue
                    
                    entry_price = float(era_h1['open'].iloc[idx+1])
                    vol = float(vol_series.iloc[idx])
                    if np.isnan(vol) or vol == 0: continue
                    
                    # Dynamic Stops (Conservative)
                    stop_dist = max(2.5, entry_price * vol * 2.0)
                    sl = entry_price - stop_dist
                    tp = entry_price + (stop_dist * 2.0)
                    
                    outcome = 0
                    
                    for i in range(idx+1, idx+25):
                        row = era_h1.iloc[i]
                        if row['low'] <= sl:
                            # Stress Slippage
                            slip = vol * entry_price * SLIP_FACTOR
                            outcome = (sl - slip - entry_price) - TOTAL_FIXED_COST
                            break
                        elif row['high'] >= tp:
                            outcome = (tp - entry_price) - TOTAL_FIXED_COST
                            break
                    
                    if outcome == 0:
                        outcome = (float(era_h1['close'].iloc[idx+24]) - entry_price) - TOTAL_FIXED_COST
                    
                    risk_amt = equity * 0.0025 # 0.25% Risk
                    units = risk_amt / stop_dist
                    trades.append(outcome * units)
                    equity += (outcome * units)
                
                # --- METRICS ---
                if not trades:
                    self.stdout.write(f"{era_name:<12} | {ema_win:<3} | {atr_mult:<3} | 0      | N/A    | N/A    | 0.0%     | NO TRADES")
                    continue
                    
                wins = sum(t for t in trades if t > 0)
                losses = abs(sum(t for t in trades if t <= 0))
                pf = wins / losses if losses > 0 else 99.9
                win_rate = len([t for t in trades if t > 0]) / len(trades)
                
                # DD
                eq = pd.Series([10000] + trades).cumsum()
                dd = (eq - eq.cummax()) / eq.cummax()
                max_dd = dd.min()
                
                # Verdict
                status = "FAIL"
                color = self.style.ERROR
                
                # STRICT CRITERIA: PF >= 1.2 under stress, MaxDD > -12%
                if pf >= 1.2 and max_dd > -0.12:
                    status = "ROBUST"
                    color = self.style.SUCCESS
                elif pf >= 1.0:
                    status = "WEAK"
                    color = self.style.WARNING
                    
                self.stdout.write(color(f"{era_name:<12} | {ema_win:<3} | {atr_mult:<3} | {len(trades):<6} | {win_rate:.1%}  | {pf:.2f}   | {max_dd:.1%}   | {status}"))


                