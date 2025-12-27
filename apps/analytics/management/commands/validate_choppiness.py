# apps/analytics/management/commands/validate_choppiness.py
import pandas as pd
import numpy as np
from django.core.management.base import BaseCommand
from apps.analytics.services import OHLCVLoader
from apps.market_data.models import Asset
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from datetime import datetime
import pytz

class Command(BaseCommand):
    help = "The Quality Audit: Choppiness Filter Validation"

    def handle(self, *args, **options):
        symbol = 'XAUUSD'
        tf = 'H1'
        
        eras = {
            "Covid_Shock":  ("2020-01-01", "2021-12-31"), 
            "Geo_Conflict": ("2022-01-01", "2024-12-18"), 
        }

        self.stdout.write(self.style.WARNING(f"--- CHOPPINESS FILTER AUDIT: {symbol} ---"))
        self.stdout.write("New Logic: Trade ONLY if CHOP < 50 (Linearity Check)")
        self.stdout.write("-" * 100)
        self.stdout.write(f"{'Era':<15} | {'Trades':<6} | {'Win%':<6} | {'PF':<6} | {'Profit':<10} | {'MaxDD':<8} | {'Status'}")
        self.stdout.write("-" * 100)

        try:
            asset = Asset.objects.get(symbol=symbol)
        except:
            self.stdout.write(self.style.ERROR("Asset not found"))
            return

        loader = OHLCVLoader()

        for era_name, (start, end) in eras.items():
            start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
            end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
            
            try:
                df = loader.load_dataframe(asset, tf, start_dt, end_dt)
            except ValueError:
                self.stdout.write(f"{era_name:<15} | NO DATA")
                continue

            if len(df) < 500:
                self.stdout.write(f"{era_name:<15} | NO DATA")
                continue

            # --- RAW LOGIC ---
            close = df['close']
            high = df['high']
            low = df['low']
            
            # Indicators
            ema_200 = EMAIndicator(close=close, window=200).ema_indicator()
            rsi = RSIIndicator(close=close, window=14).rsi()
            adx = ADXIndicator(high=high, low=low, close=close, window=14).adx()
            
            # Manual Chop
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            tr_sum = tr.rolling(14).sum()
            range_hl = high.rolling(14).max() - low.rolling(14).min()
            chop = 100 * np.log10(tr_sum / range_hl.replace(0, np.nan)) / np.log10(14)
            chop.fillna(100, inplace=True) # Assume chop if data missing
            
            # Signal
            signals = (close > ema_200) & \
                      (rsi < 45) & \
                      (adx > 20) & \
                      (chop < 50) & \
                      (close > df['open'])
            
            # --- SIMULATION ---
            trades = []
            equity = 10000.0
            COSTS = 0.50 
            
            indices = np.where(signals)[0]
            vol_series = close.pct_change().rolling(20).std()
            
            for idx in indices:
                if idx + 24 >= len(df): continue
                
                entry_price = float(df['open'].iloc[idx+1])
                vol = float(vol_series.iloc[idx])
                if np.isnan(vol) or vol == 0: continue
                
                stop_dist = max(2.0, entry_price * vol * 2.0)
                sl = entry_price - stop_dist
                tp = entry_price + (stop_dist * 2.0)
                
                outcome_pnl = 0
                
                for i in range(idx+1, idx+25):
                    row = df.iloc[i]
                    if row['low'] <= sl:
                        outcome_pnl = (sl - entry_price) - COSTS
                        break
                    elif row['high'] >= tp:
                        outcome_pnl = (tp - entry_price) - COSTS
                        break
                
                if outcome_pnl == 0:
                    outcome_pnl = (float(df['close'].iloc[idx+24]) - entry_price) - COSTS
                
                risk_amt = equity * 0.005
                units = risk_amt / stop_dist
                pnl = outcome_pnl * units
                
                trades.append(pnl)
                equity += pnl

            # --- METRICS ---
            if not trades:
                self.stdout.write(f"{era_name:<15} | 0      | N/A    | N/A    | $0         | 0.0%     | NO TRADES")
                continue
                
            wins = sum(1 for t in trades if t > 0)
            losses = sum(1 for t in trades if t <= 0)
            gross_win = sum(t for t in trades if t > 0)
            gross_loss = abs(sum(t for t in trades if t <= 0))
            
            pf = gross_win / gross_loss if gross_loss > 0 else 99.9
            net = sum(trades)
            
            eq_curve = pd.Series([10000] + trades).cumsum()
            dd = (eq_curve - eq_curve.cummax()) / eq_curve.cummax()
            max_dd = dd.min()
            
            status = "FAIL"
            if era_name == "Covid_Shock":
                if pf >= 0.9 and max_dd > -0.15: status = "SURVIVED"
            elif pf >= 1.3:
                status = "PASS"
            
            color = self.style.SUCCESS if status in ["PASS", "SURVIVED"] else self.style.ERROR
            self.stdout.write(color(f"{era_name:<15} | {len(trades):<6} | {wins/len(trades):.1%}  | {pf:.2f}   | ${net:<9.0f} | {max_dd:.1%}   | {status}"))

            