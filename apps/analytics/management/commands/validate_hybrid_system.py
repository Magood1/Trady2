# apps/analytics/management/commands/validate_hybrid_system.py
import pandas as pd
import numpy as np
from django.core.management.base import BaseCommand
from apps.analytics.services import OHLCVLoader
from apps.market_data.models import Asset
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator
from datetime import datetime
import pytz

class Command(BaseCommand):
    help = "The Chimera Test: Adaptive Hybrid Strategy (Trend + Mean Rev)"

    def handle(self, *args, **options):
        symbol = 'XAUUSD'
        tf = 'H1'
        
        eras = {
            "Boring_Era":   ("2018-01-01", "2019-12-31"), # Range
            "Covid_Shock":  ("2020-01-01", "2021-12-31"), # Volatility
            "Geo_Conflict": ("2022-01-01", "2024-12-18"), # Trend
        }

        self.stdout.write(self.style.WARNING("--- CHIMERA HYBRID SYSTEM STRESS TEST ---"))
        self.stdout.write("Logic: Switch between Trend/Range based on ADX(14) threshold 25")
        self.stdout.write("-" * 110)
        self.stdout.write(f"{'Era':<15} | {'Trades':<6} | {'Win%':<6} | {'Net PF':<6} | {'Profit':<10} | {'MaxDD':<8} | {'Status'}")
        self.stdout.write("-" * 110)

        asset = Asset.objects.get(symbol=symbol)
        loader = OHLCVLoader()

        for era_name, (start, end) in eras.items():
            start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
            end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
            
            df = loader.load_dataframe(asset, tf, start_dt, end_dt)
            if len(df) < 500:
                self.stdout.write(f"{era_name:<15} | NO DATA")
                continue

            # --- INDICATORS ---
            close = df['close']
            high = df['high']
            low = df['low']
            
            # Regime
            adx = ADXIndicator(high, low, close, window=14).adx()
            
            # Trend Tools
            ema_200 = EMAIndicator(close, window=200).ema_indicator()
            
            # Range Tools (Wider Bands for safety)
            bb = BollingerBands(close, window=20, window_dev=2.5)
            
            # Trigger
            rsi = RSIIndicator(close, window=14).rsi()
            
            # --- HYBRID LOGIC ---
            # 1. Trend Mode (ADX > 25) -> Only Longs for Gold (Bias)
            trend_setup = (adx > 25) & (close > ema_200)
            trend_signal = trend_setup & (rsi < 50) & (close > df['open']) # Pullback Entry
            
            # 2. Range Mode (ADX < 25) -> Long & Short
            range_setup = (adx < 25)
            # Buy Low
            range_buy = range_setup & (close < bb.bollinger_lband()) & (rsi < 30)
            # Sell High
            range_sell = range_setup & (close > bb.bollinger_hband()) & (rsi > 70)
            
            # Simulation
            trades = []
            equity = 10000.0
            COSTS = 0.50 
            
            # Volatility for stops
            vol_series = close.pct_change().rolling(20).std()
            
            # Iterate
            # We skip first 200 bars
            for i in range(200, len(df) - 24):
                # Check signals at i
                idx = df.index[i]
                
                is_trend_buy = trend_signal.iloc[i]
                is_range_buy = range_buy.iloc[i]
                is_range_sell = range_sell.iloc[i]
                
                if not (is_trend_buy or is_range_buy or is_range_sell):
                    continue
                    
                entry_price = float(df['open'].iloc[i+1])
                vol = float(vol_series.iloc[i])
                if np.isnan(vol) or vol == 0: continue
                
                outcome_pnl = 0
                signal_type = "NONE"
                
                # --- EXECUTION LOGIC (ADAPTIVE) ---
                
                if is_trend_buy:
                    signal_type = "TREND_BUY"
                    # Trend Rules: Wide Stop, Big Target
                    stop_dist = max(2.0, entry_price * vol * 2.0)
                    sl = entry_price - stop_dist
                    tp = entry_price + (stop_dist * 2.0) # 2R
                    
                    # Trend Loop
                    for j in range(i+1, i+49): # Allow 48 hrs for trend
                        curr_low = float(low.iloc[j])
                        curr_high = float(high.iloc[j])
                        if curr_low <= sl:
                            outcome_pnl = (sl - entry_price) - COSTS
                            break
                        elif curr_high >= tp:
                            outcome_pnl = (tp - entry_price) - COSTS
                            break
                            
                elif is_range_buy:
                    signal_type = "RANGE_BUY"
                    # Range Rules: Tight Stop, Target Mean
                    stop_dist = max(2.0, entry_price * vol * 3.0) # Wide stop for wicks
                    sl = entry_price - stop_dist
                    tp_dynamic = float(bb.bollinger_mavg().iloc[i]) # Target SMA
                    if tp_dynamic <= entry_price: tp_dynamic = entry_price + 2.0 # Minimum target
                    
                    for j in range(i+1, i+25): # 24 hrs max
                        curr_low = float(low.iloc[j])
                        curr_high = float(high.iloc[j])
                        curr_sma = float(bb.bollinger_mavg().iloc[j])
                        
                        # TP is dynamic (SMA)
                        if curr_high >= curr_sma:
                            outcome_pnl = (curr_sma - entry_price) - COSTS
                            break
                        if curr_low <= sl:
                            outcome_pnl = (sl - entry_price) - COSTS
                            break

                elif is_range_sell:
                    signal_type = "RANGE_SELL"
                    stop_dist = max(2.0, entry_price * vol * 3.0)
                    sl = entry_price + stop_dist
                    tp_dynamic = float(bb.bollinger_mavg().iloc[i])
                    if tp_dynamic >= entry_price: tp_dynamic = entry_price - 2.0
                    
                    for j in range(i+1, i+25):
                        curr_low = float(low.iloc[j])
                        curr_high = float(high.iloc[j])
                        curr_sma = float(bb.bollinger_mavg().iloc[j])
                        
                        if curr_low <= curr_sma:
                            outcome_pnl = (entry_price - curr_sma) - COSTS
                            break
                        if curr_high >= sl:
                            outcome_pnl = (entry_price - sl) - COSTS
                            break
                
                # Time Exit
                if outcome_pnl == 0:
                    exit_p = float(close.iloc[i+24])
                    if "BUY" in signal_type:
                        outcome_pnl = (exit_p - entry_price) - COSTS
                    else:
                        outcome_pnl = (entry_price - exit_p) - COSTS
                
                # Skip consecutive signals (Simplification)
                # In real code, we'd manage state. Here we just take them all (Stress test)
                
                # Risk 0.5%
                risk_amt = equity * 0.005
                # Approx units
                stop_d = max(2.0, entry_price * vol * 2.0)
                units = risk_amt / stop_d
                
                pnl_usd = outcome_pnl * units
                trades.append(pnl_usd)
                equity += pnl_usd

            # Metrics
            if not trades:
                self.stdout.write(f"{era_name:<15} | 0      | N/A")
                continue
                
            wins = sum(1 for t in trades if t > 0)
            losses = sum(1 for t in trades if t <= 0)
            gross_win = sum(t for t in trades if t > 0)
            gross_loss = abs(sum(t for t in trades if t <= 0))
            
            pf = gross_win / gross_loss if gross_loss > 0 else 99.9
            win_rate = wins / len(trades)
            net = sum(trades)
            
            # Max DD
            eq_curve = pd.Series([10000] + trades).cumsum()
            dd = (eq_curve - eq_curve.cummax()) / eq_curve.cummax()
            max_dd = dd.min()
            
            status = "FAIL"
            if pf >= 1.3 and max_dd > -0.15: status = "PASS"
            elif pf >= 1.1: status = "WEAK"
            
            color = self.style.SUCCESS if status == "PASS" else self.style.ERROR
            self.stdout.write(color(f"{era_name:<15} | {len(trades):<6} | {win_rate:.1%}  | {pf:.2f}   | ${net:<9.0f} | {max_dd:.1%}   | {status}"))
            