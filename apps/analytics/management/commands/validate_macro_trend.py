# apps/analytics/management/commands/validate_macro_trend.py
import pandas as pd
import numpy as np
from django.core.management.base import BaseCommand
from apps.analytics.services import OHLCVLoader
from apps.market_data.models import Asset
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator
from datetime import datetime
import pytz

class Command(BaseCommand):
    help = "The Final Pivot: H4/D1 Macro Trend Following (Long Only)"

    def handle(self, *args, **options):
        symbol = 'XAUUSD'
        
        eras = {
            "Gold_Crash":   ("2013-01-01", "2013-12-31"), # Bear Market (Should stay out)
            "Boring_Era":   ("2018-01-01", "2019-12-31"), # Range
            "Covid_Shock":  ("2020-01-01", "2021-12-31"), # Volatility
            "Geo_Conflict": ("2022-01-01", "2024-12-18"), # Bull Run
        }

        self.stdout.write(self.style.WARNING("--- MACRO SNIPER TEST (H4/D1 Long Only) ---"))
        self.stdout.write("Strategy: D1 Uptrend Filter + H4 Dip Buy + Wide Trailing Stop")
        self.stdout.write("-" * 100)
        self.stdout.write(f"{'Era':<15} | {'Trades':<6} | {'Win%':<6} | {'Net PF':<6} | {'Profit':<10} | {'MaxDD':<8} | {'Status'}")
        self.stdout.write("-" * 100)

        asset = Asset.objects.get(symbol=symbol)
        loader = OHLCVLoader()

        for era_name, (start, end) in eras.items():
            start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
            end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
            
            # Load Data
            # We need H4 primarily. We will simulate D1 EMA using H4 resampled or approx (6 * H4 bars)
            df = loader.load_dataframe(asset, 'H4', start_dt, end_dt)
            
            if len(df) < 200:
                self.stdout.write(f"{era_name:<15} | NO DATA")
                continue

            # --- INDICATORS ---
            close = df['close']
            
            # 1. Macro Trend Filter (Approx D1 EMA 200 on H4 data = EMA 1200)
            # 200 days * 6 candles/day = 1200 bars
            ema_macro = EMAIndicator(close, window=1200).ema_indicator()
            
            # 2. Dip Buying (H4 Bollinger Lower Band)
            bb = BollingerBands(close, window=50, window_dev=2.0)
            
            # --- LOGIC ---
            # Buy ONLY if Price > Macro EMA (Bull Market) AND Price < Lower Band (Dip)
            # AND Green Candle Confirmation
            signals = (close > ema_macro) & \
                      (close < bb.bollinger_lband()) & \
                      (close > df['open'])
            
            # Execution
            trades = []
            equity = 10000.0
            
            # SWAP Cost per trade (approx holding 3 days * $5)
            # Spread (Macro spread is negligible relative to target)
            FIXED_COST = 20.0 # High cost estimate to include swap/spread
            
            indices = np.where(signals)[0]
            
            # State
            in_position = False
            entry_price = 0
            stop_loss = 0
            
            # Volatility
            atr = (df['high'] - df['low']).rolling(50).mean()
            
            for i in range(len(df)):
                # Logic to enter/exit
                # Simplified loop for speed
                if i < 1200: continue
                
                curr_price = float(close.iloc[i])
                curr_low = float(df['low'].iloc[i])
                curr_atr = float(atr.iloc[i])
                
                if in_position:
                    # Exit Logic: Trailing Stop
                    # Move SL to breakeven after 1 ATR profit
                    if (curr_price - entry_price) > curr_atr:
                        stop_loss = max(stop_loss, entry_price) # Move to BE
                    
                    # Trailing: Keep SL 2 ATR below high
                    new_sl = curr_price - (curr_atr * 2.0)
                    stop_loss = max(stop_loss, new_sl)
                    
                    # Check Hit
                    if curr_low <= stop_loss:
                        pnl = (stop_loss - entry_price) * units - FIXED_COST
                        equity += pnl
                        trades.append(pnl)
                        in_position = False
                        
                else:
                    # Check Entry
                    if signals.iloc[i]:
                        entry_price = float(df['open'].iloc[i+1]) if i+1 < len(df) else curr_price
                        stop_loss = entry_price - (curr_atr * 3.0) # Wide initial stop
                        
                        # Risk Management: 1% Risk
                        risk_amt = equity * 0.01
                        dist = entry_price - stop_loss
                        if dist <= 0: continue
                        units = risk_amt / dist
                        
                        in_position = True

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
            
            # Approx MaxDD
            eq_curve = pd.Series([10000] + trades).cumsum()
            dd = (eq_curve - eq_curve.cummax()) / eq_curve.cummax()
            max_dd = dd.min()
            
            status = "FAIL"
            if pf >= 2.0: status = "HOLY GRAIL"
            elif pf >= 1.5: status = "EXCELLENT"
            elif pf >= 1.3: status = "PASS"
            
            color = self.style.SUCCESS if status in ["PASS", "EXCELLENT", "HOLY GRAIL"] else self.style.ERROR
            self.stdout.write(color(f"{era_name:<15} | {len(trades):<6} | {win_rate:.1%}  | {pf:.2f}   | ${net:<9.0f} | {max_dd:.1%}   | {status}"))

            