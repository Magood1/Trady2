# apps/analytics/management/commands/run_century_fractal.py
import pandas as pd
import numpy as np
from django.core.management.base import BaseCommand
from apps.market_data.services import ingest_ohlcv_data
from apps.market_data.models import OHLCV, Asset
from apps.common.enums import Timeframe
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from datetime import datetime, timedelta
import pytz
import time

class Command(BaseCommand):
    help = "The Century Test: Robust H1 Fractal Strategy (2015-2024)"

    def handle(self, *args, **options):
        symbol = 'XAUUSD'
        tf_str = 'H1'
        
        # Focus on eras where we confirmed data availability in previous tests
        eras = {
            "QE_Era":       ("2015-01-01", "2016-12-31"), 
            "Boring_Era":   ("2018-01-01", "2019-12-31"),
            "Covid_Shock":  ("2020-01-01", "2021-12-31"),
            "Infl_Crisis":  ("2022-01-01", "2023-12-31"),
            "Geo_Conflict": ("2024-01-01", "2024-12-18"),
        }

        self.stdout.write(self.style.ERROR(f"--- ROBUST FRACTAL TEST: {symbol} (H1 Only) ---"))
        self.stdout.write("Strategy: H1 Trend (EMA200, ADX>20) + H1 Pullback Entry (RSI<45)")
        self.stdout.write("-" * 100)
        self.stdout.write(f"{'Era':<15} | {'Trades':<6} | {'Win%':<6} | {'Net PF':<6} | {'Profit':<10} | {'MaxDD':<8} | {'Status'}")
        self.stdout.write("-" * 100)

        for era_name, (start, end) in eras.items():
            self.run_era_test(symbol, tf_str, era_name, start, end)

    def run_era_test(self, symbol, tf_str, era_name, start_str, end_str):
        start_dt = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
        end_dt = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=pytz.UTC)

        # 1. Ingest Data (Robust)
        try:
            ingest_ohlcv_data(symbol, Timeframe(tf_str), start_dt, end_dt)
            time.sleep(1) # Allow DB commit
        except Exception:
            pass 

        # 2. Load Data
        qs = OHLCV.objects.filter(
            asset__symbol=symbol, 
            timeframe=tf_str, 
            timestamp__range=(start_dt, end_dt)
        ).order_by('timestamp')
        
        df = pd.DataFrame.from_records(qs.values('timestamp', 'open', 'high', 'low', 'close'))

        if len(df) < 500:
            self.stdout.write(f"{era_name:<15} | NO DATA (Found {len(df)})")
            return

        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df.set_index('timestamp', inplace=True)
        for c in ['open', 'high', 'low', 'close']: df[c] = pd.to_numeric(df[c])

        # 3. Strategy Logic (H1 as the Fractal Unit)
        # Using H1 to emulate the "Macro" trend and "Micro" entry simultaneously
        
        # Macro Trend Proxy: Price > EMA 200
        ema_200 = EMAIndicator(close=df['close'], window=200).ema_indicator()
        adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()
        
        # Micro Entry Proxy: RSI Pullback
        rsi = RSIIndicator(close=df['close'], window=14).rsi()
        
        # Logic:
        # 1. Trend: Close > EMA 200 (Bullish Context)
        # 2. Strength: ADX > 20 (Avoid chop)
        # 3. Entry: RSI < 45 (Buying the dip)
        # 4. Confirm: Close > Open (Green Candle - Reversal sign)
        signals = (df['close'] > ema_200) & (adx > 20) & (rsi < 45) & (df['close'] > df['open'])
        
        # 4. Simulation Loop
        COSTS = 0.50 # Spread + Comm + Slip (~$0.50 per unit impact on Gold)
        
        indices = np.where(signals)[0]
        trades = []
        equity_curve = [10000.0]
        
        # Volatility for stops
        df['vol'] = df['close'].pct_change().rolling(20).std()

        for idx in indices:
            if idx + 24 >= len(df): continue # 24H horizon
            
            entry_price = float(df['open'].iloc[idx+1])
            vol = float(df['vol'].iloc[idx])
            if np.isnan(vol) or vol == 0: continue

            # Dynamic Risk Management (Validated in Trinity Test)
            stop_dist = max(2.0, entry_price * vol * 2.0) # 2.0x Vol Stop
            sl_price = entry_price - stop_dist
            tp_price = entry_price + (stop_dist * 2.0) # 2.0 Reward
            
            outcome_pnl = 0
            
            # Outcome Loop
            for i in range(idx+1, idx+25):
                row = df.iloc[i]
                low = float(row['low'])
                high = float(row['high'])
                
                if low <= sl_price:
                    # Stop Hit
                    slip = vol * entry_price * 0.1
                    outcome_pnl = (sl_price - slip - entry_price) - COSTS
                    break
                elif high >= tp_price:
                    # Target Hit
                    outcome_pnl = (tp_price - entry_price) - COSTS
                    break
            
            if outcome_pnl == 0: # Time Exit
                exit_p = float(df['close'].iloc[idx+24])
                outcome_pnl = (exit_p - entry_price) - COSTS
            
            # Position Sizing (0.5% Risk)
            risk_amt = equity_curve[-1] * 0.005
            units = risk_amt / stop_dist
            pnl_usd = outcome_pnl * units
            
            equity_curve.append(equity_curve[-1] + pnl_usd)
            trades.append(pnl_usd)

        # 5. Metrics
        if not trades:
            self.stdout.write(f"{era_name:<15} | 0      | N/A    | N/A    | $0         | 0.0%     | NO TRADES")
            return

        wins = sum(1 for t in trades if t > 0)
        losses = sum(1 for t in trades if t <= 0)
        gross_win = sum(t for t in trades if t > 0)
        gross_loss = abs(sum(t for t in trades if t <= 0))
        
        win_rate = wins / len(trades)
        pf = gross_win / gross_loss if gross_loss > 0 else 99.9
        net_profit = sum(trades)
        
        # Max DD
        eq = pd.Series(equity_curve)
        dd = (eq - eq.cummax()) / eq.cummax()
        max_dd = dd.min()
        
        # Status Logic (Lenient for H1)
        status = "FAIL"
        if pf >= 1.3: status = "EXCELLENT"
        elif pf >= 1.1 and max_dd > -0.10: status = "PASS"
        elif pf >= 1.0: status = "WEAK"
        
        color = self.style.SUCCESS if status in ["PASS", "EXCELLENT"] else self.style.ERROR
        self.stdout.write(color(
            f"{era_name:<15} | {len(trades):<6} | {win_rate:.1%}  | {pf:.2f}   | ${net_profit:<9.0f} | {max_dd:.1%}   | {status}"
        ))
        