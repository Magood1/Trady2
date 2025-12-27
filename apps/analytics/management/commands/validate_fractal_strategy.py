# apps/analytics/management/commands/validate_fractal_strategy.py
import pandas as pd
import numpy as np
from django.core.management.base import BaseCommand
from apps.analytics.services import OHLCVLoader
from apps.market_data.models import Asset
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from datetime import datetime, timezone

class Command(BaseCommand):
    help = "Fractal Strategy Test: H4 Trend + M15 Entry"

    def handle(self, *args, **options):
        symbol = 'XAUUSD'
        
        self.stdout.write(self.style.WARNING("--- FRACTAL STRATEGY STRESS TEST (MTF) ---"))
        self.stdout.write("Logic: H4 Trend (EMA50 > EMA200 + ADX>25) + M15 Pullback (RSI<30)")
        self.stdout.write("-" * 90)
        self.stdout.write(f"{'Period':<8} | {'Trades':<6} | {'Win%':<8} | {'Net PF':<6} | {'Profit':<10} | {'Status'}")
        self.stdout.write("-" * 90)

        # 1. Load Data
        asset = Asset.objects.get(symbol=symbol)
        loader = OHLCVLoader()
        
        # Load Macro (H4) & Micro (M15)
        # Load ample history to allow indicators to warm up
        df_h4 = loader.load_dataframe(asset, 'H4', pd.Timestamp.min.tz_localize('UTC'), pd.Timestamp.max.tz_localize('UTC'))
        df_m15 = loader.load_dataframe(asset, 'M15', pd.Timestamp.min.tz_localize('UTC'), pd.Timestamp.max.tz_localize('UTC'))
        
        if df_h4.empty or df_m15.empty:
            self.stdout.write(self.style.ERROR("Missing Data for H4 or M15"))
            return

        # 2. Macro Logic (H4)
        # Trend Definition: Strong Uptrend
        h4_ema50 = EMAIndicator(close=df_h4['close'], window=50).ema_indicator()
        h4_ema200 = EMAIndicator(close=df_h4['close'], window=200).ema_indicator()
        h4_adx = ADXIndicator(high=df_h4['high'], low=df_h4['low'], close=df_h4['close'], window=14).adx()
        
        # 1 = Uptrend, 0 = No Trend
        df_h4['macro_trend'] = ((h4_ema50 > h4_ema200) & (h4_adx > 25)).astype(int)
        
        # Shift H4 by 1 candle to avoid look-ahead bias!
        # (We only know H4 trend AFTER the candle closes)
        df_h4['macro_trend'] = df_h4['macro_trend'].shift(1)
        
        # 3. Merge MTF (The Critical Step)
        # Resample H4 to match M15 index, forward filling the trend status
        # This aligns the "Closed H4 Trend" to "Current M15 Candle"
        macro_aligned = df_h4['macro_trend'].reindex(df_m15.index, method='ffill')
        df_m15['macro_trend'] = macro_aligned.fillna(0)

        # 4. Micro Logic (M15)
        # Entry: Pullback in Uptrend
        m15_rsi = RSIIndicator(close=df_m15['close'], window=14).rsi()
        
        # Signal: Macro Uptrend AND Micro Oversold
        signals = (df_m15['macro_trend'] == 1) & (m15_rsi < 30) & (df_m15['close'] > df_m15['open']) # Green confirmation
        
        # 5. Financial Simulation (Strict)
        # Costs for Gold (M15 requires tighter spreads, so impact is higher)
        SPREAD = 0.30
        COMMISSION = 0.10
        SLIPPAGE_FACTOR = 0.1 # 10% of Volatility
        
        # Filter for 2024 Analysis
        start_2024 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        sim_df = df_m15[df_m15.index >= start_2024].copy()
        sim_signals = signals[signals.index >= start_2024]
        
        # Volatility for stops (M15 Volatility)
        sim_df['vol'] = sim_df['close'].pct_change().rolling(20).std()
        
        indices = np.where(sim_signals)[0]
        
        trades = []
        equity = 10000.0
        
        for idx in indices:
            if idx + 48 >= len(sim_df): continue # 48 bars (12 hours on M15) time limit
            
            entry_price = float(sim_df['open'].iloc[idx+1])
            vol = float(sim_df['vol'].iloc[idx])
            if np.isnan(vol) or vol == 0: continue

            # Dynamic Stops (Adjusted for M15 noise)
            # M15 needs wider breathing room relative to vol
            stop_dist = max(1.5, entry_price * vol * 3.0) 
            sl_price = entry_price - stop_dist
            tp_price = entry_price + (stop_dist * 2.0) # 2.0 Reward
            
            outcome_pnl = 0
            
            # Fast Loop (12 Hours max hold)
            for i in range(idx+1, idx+49):
                row = sim_df.iloc[i]
                low = float(row['low'])
                high = float(row['high'])
                
                if low <= sl_price:
                    slip = vol * entry_price * SLIPPAGE_FACTOR
                    outcome_pnl = (sl_price - slip) - entry_price - (SPREAD + COMMISSION)
                    break
                elif high >= tp_price:
                    outcome_pnl = tp_price - entry_price - (SPREAD + COMMISSION)
                    break
            
            if outcome_pnl == 0: # Time Exit
                exit_p = float(sim_df['close'].iloc[idx+48])
                outcome_pnl = exit_p - entry_price - (SPREAD + COMMISSION)
            
            # 0.5% Risk
            risk_amt = equity * 0.005
            units = risk_amt / stop_dist
            trades.append({'date': sim_df.index[idx], 'pnl': outcome_pnl * units})
            equity += (outcome_pnl * units)

        # 6. Reporting
        if not trades:
            self.stdout.write("No trades generated.")
            return

        df_t = pd.DataFrame(trades)
        df_t['quarter'] = df_t['date'].dt.to_period('Q')
        
        valid_quarters = 0
        for q in sorted(df_t['quarter'].unique()):
            q_data = df_t[df_t['quarter'] == q]
            if len(q_data) == 0: continue
            
            wins = q_data[q_data['pnl'] > 0]['pnl'].sum()
            losses = abs(q_data[q_data['pnl'] <= 0]['pnl'].sum())
            pf = wins / losses if losses > 0 else 99.9
            win_rate = (q_data['pnl'] > 0).mean()
            
            status = "FAIL"
            if pf >= 1.3 and len(q_data) > 15:
                status = "PASS"
                valid_quarters += 1
            elif pf >= 1.1:
                status = "WEAK"
                
            color = self.style.SUCCESS if status == "PASS" else self.style.ERROR
            self.stdout.write(color(f"{str(q):<8} | {len(q_data):<6} | {win_rate:.1%}    | {pf:.2f}   | ${q_data['pnl'].sum():.0f}      | {status}"))

        if valid_quarters >= 3: # We want consistency across 3 quarters for MTF
            self.stdout.write(self.style.SUCCESS("\n*** FRACTAL STRATEGY: APPROVED ***"))
        else:
            self.stdout.write(self.style.ERROR("\n*** FRACTAL STRATEGY: FAILED ***"))
            