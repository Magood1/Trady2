# apps/analytics/management/commands/validate_regime_gated.py
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
    help = "The Omega Test: Volatility-Gated Fractal Strategy (Robust)"

    def handle(self, *args, **options):
        symbol = 'XAUUSD'
        
        eras = {
            "Boring_Era":   ("2018-01-01", "2019-12-31"), # The Killer Period
            "Covid_Shock":  ("2020-01-01", "2021-12-31"), # High Vol
            "Geo_Conflict": ("2022-01-01", "2024-12-18"), # Trend
        }

        self.stdout.write(self.style.WARNING(f"--- OMEGA TEST: REGIME GATED STRATEGY ({symbol}) ---"))
        self.stdout.write("Logic: Trade H1 Trend ONLY if Daily ATR > $20 (High Volatility Regime)")
        self.stdout.write("-" * 110)
        self.stdout.write(f"{'Era':<15} | {'Trades':<6} | {'Win%':<6} | {'PF':<6} | {'Profit':<10} | {'MaxDD':<8} | {'Status'}")
        self.stdout.write("-" * 110)

        try:
            asset = Asset.objects.get(symbol=symbol)
        except Asset.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"Asset {symbol} not found."))
            return

        loader = OHLCVLoader()

        for era_name, (start, end) in eras.items():
            start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
            end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
            
            # 1. Fault-Tolerant Data Loading
            try:
                # Load D1 first (The Gate)
                df_d1 = loader.load_dataframe(asset, 'D1', start_dt, end_dt)
                if len(df_d1) < 50: raise ValueError("Not enough D1 data")
                
                # Load H1 (The Signal)
                df_h1 = loader.load_dataframe(asset, 'H1', start_dt, end_dt)
                if len(df_h1) < 200: raise ValueError("Not enough H1 data")
                
            except ValueError:
                # Graceful skip if data is missing (common for old eras)
                self.stdout.write(f"{era_name:<15} | NO DATA (Broker Limit)")
                continue

            # --- 2. MACRO REGIME (D1) ---
            # Calculate Daily ATR
            d1_atr = AverageTrueRange(high=df_d1['high'], low=df_d1['low'], close=df_d1['close'], window=14).average_true_range()
            
            # Define Regime: 1 if ATR > 20 (Gold moving > $20/day), else 0
            # Shift 1 day to avoid lookahead (We trade today based on yesterday's volatility state)
            df_d1['is_active_regime'] = (d1_atr > 20.0).astype(int).shift(1)
            
            # Align to H1 (Forward fill the daily state to hourly bars)
            regime_aligned = df_d1['is_active_regime'].reindex(df_h1.index, method='ffill').fillna(0)
            df_h1['regime_on'] = regime_aligned

            # --- 3. TREND FILTER (H1) ---
            h1_ema200 = EMAIndicator(close=df_h1['close'], window=200).ema_indicator()
            h1_adx = ADXIndicator(high=df_h1['high'], low=df_h1['low'], close=df_h1['close'], window=14).adx()
            h1_rsi = RSIIndicator(close=df_h1['close'], window=14).rsi()
            
            # --- 4. SIGNAL LOGIC ---
            signals = (df_h1['regime_on'] == 1) & \
                      (df_h1['close'] > h1_ema200) & \
                      (h1_adx > 25) & \
                      (h1_rsi < 45) & \
                      (df_h1['close'] > df_h1['open'])
            
            # --- 5. SIMULATION ---
            trades = []
            equity = 10000.0
            COSTS = 0.50 
            
            indices = np.where(signals)[0]
            
            # Volatility for stops (H1 ATR Proxy)
            vol_series = df_h1['close'].pct_change().rolling(20).std()
            
            for idx in indices:
                if idx + 24 >= len(df_h1): continue
                
                entry_price = float(df_h1['open'].iloc[idx+1])
                vol = float(vol_series.iloc[idx])
                if np.isnan(vol) or vol == 0: continue
                
                # Dynamic Risk
                stop_dist = max(2.0, entry_price * vol * 2.0)
                sl = entry_price - stop_dist
                tp = entry_price + (stop_dist * 2.0) 
                
                outcome_pnl = 0
                
                for i in range(idx+1, idx+25):
                    row = df_h1.iloc[i]
                    if row['low'] <= sl:
                        outcome_pnl = (sl - entry_price) - COSTS
                        break
                    elif row['high'] >= tp:
                        outcome_pnl = (tp - entry_price) - COSTS
                        break
                
                if outcome_pnl == 0: # Time Exit
                    outcome_pnl = (float(df_h1['close'].iloc[idx+24]) - entry_price) - COSTS
                
                # 0.5% Risk Sizing
                risk_amt = equity * 0.005
                units = risk_amt / stop_dist
                pnl = outcome_pnl * units
                
                trades.append(pnl)
                equity += pnl

            # --- 6. METRICS ---
            if not trades:
                # If regime filter worked perfectly (0 trades in bad era), it's a PASS
                status = "PASS (SAFE)" if era_name == "Boring_Era" else "NO TRADES"
                self.stdout.write(f"{era_name:<15} | 0      | N/A    | N/A    | $0         | 0.0%     | {status}")
                continue
                
            wins = sum(1 for t in trades if t > 0)
            losses = sum(1 for t in trades if t <= 0)
            gross_win = sum(t for t in trades if t > 0)
            gross_loss = abs(sum(t for t in trades if t <= 0))
            
            pf = gross_win / gross_loss if gross_loss > 0 else 99.9
            net = sum(trades)
            
            # Max DD
            eq_series = pd.Series([10000] + trades).cumsum()
            dd = (eq_series - eq_series.cummax()) / eq_series.cummax()
            max_dd = dd.min()
            
            status = "FAIL"
            if pf >= 1.3 and max_dd > -0.10: status = "PASS"
            elif pf >= 1.1: status = "WEAK"
            
            color = self.style.SUCCESS if status == "PASS" else self.style.ERROR
            self.stdout.write(color(f"{era_name:<15} | {len(trades):<6} | {wins/len(trades):.1%}  | {pf:.2f}   | ${net:<9.0f} | {max_dd:.1%}   | {status}"))

            