# apps/analytics/management/commands/test_mean_reversion.py
import pandas as pd
import numpy as np
from django.core.management.base import BaseCommand
from apps.analytics.services import OHLCVLoader
from apps.market_data.models import Asset
from apps.common.enums import Timeframe
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator
from datetime import datetime
import pytz

class Command(BaseCommand):
    help = "The Fortress Test: Adaptive Mean Reversion Strategy (2018-2024)"

    def add_arguments(self, parser):
        parser.add_argument('--symbol', type=str, default='XAUUSD')
        parser.add_argument('--timeframe', type=str, default='H1')

    def handle(self, *args, **options):
        symbol = options['symbol']
        tf = options['timeframe']
        
        # Testing the Eras where Trend Following Failed
        eras = {
            "Boring_Era":   ("2018-01-01", "2019-12-31"), # Low Volatility (Trend Failed here)
            "Covid_Shock":  ("2020-01-01", "2021-12-31"), # High Volatility Chop
            "Geo_Conflict": ("2022-01-01", "2024-12-18"), # Trending (Check if MR survives/filters out)
        }

        self.stdout.write(self.style.WARNING("--- MEAN REVERSION FORTRESS TEST ---"))
        self.stdout.write("Strategy: Fade Bollinger Extremes (2.5 SD) | Filter: ADX < 30")
        self.stdout.write("-" * 100)
        self.stdout.write(f"{'Era':<15} | {'Trades':<6} | {'Win%':<6} | {'Net PF':<6} | {'Profit':<10} | {'Status'}")
        self.stdout.write("-" * 100)

        try:
            asset = Asset.objects.get(symbol=symbol)
        except Asset.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"Asset {symbol} not found."))
            return

        loader = OHLCVLoader()

        for era_name, (start, end) in eras.items():
            start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
            end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
            
            # Load Data
            try:
                df = loader.load_dataframe(asset, tf, start_dt, end_dt)
            except ValueError:
                self.stdout.write(f"{era_name:<15} | NO DATA (Loader Error)")
                continue
                
            if len(df) < 500:
                self.stdout.write(f"{era_name:<15} | NO DATA (Found {len(df)})")
                continue

            # --- STRATEGY LOGIC ---
            close = df['close']
            
            # 1. Bollinger Bands (2.5 std dev - Extreme moves only)
            bb = BollingerBands(close=close, window=20, window_dev=2.5)
            
            # 2. RSI (Extremes)
            rsi = RSIIndicator(close=close, window=14).rsi()
            
            # 3. Regime Filter (The Guard): Only trade when trend is WEAK
            # ADX < 30 means "Choppy/Ranging"
            adx = ADXIndicator(high=df['high'], low=df['low'], close=close, window=14).adx()
            
            # Entry Signals
            # Buy: Price < Lower Band & RSI < 30 & ADX < 30 (Ranging Market)
            buy_signal = (close < bb.bollinger_lband()) & (rsi < 30) & (adx < 30)
            
            # Sell: Price > Upper Band & RSI > 70 & ADX < 30
            sell_signal = (close > bb.bollinger_hband()) & (rsi > 70) & (adx < 30)
            
            # Simulation Loop
            trades = []
            equity = 10000.0
            COSTS = 0.50 # Conservative Gold Costs (Spread+Comm+Slip)
            
            # Get signal indices
            signals = buy_signal | sell_signal
            indices = np.where(signals)[0]
            
            for idx in indices:
                # Horizon check
                if idx + 24 >= len(df): continue
                
                is_buy = buy_signal.iloc[idx]
                entry_price = float(df['open'].iloc[idx+1])
                
                # Volatility for stops
                vol = float(close.pct_change().rolling(20).std().iloc[idx])
                if np.isnan(vol) or vol == 0: continue
                
                outcome_pnl = 0
                
                # Check next 24 bars
                for i in range(idx+1, idx+25):
                    curr_close = float(close.iloc[i])
                    # Mean Reversion Target: The Middle Band (SMA 20)
                    # Note: Using future SMA implies we update it bar by bar. 
                    # Approximation: Use current bar's SMA
                    sma = float(bb.bollinger_mavg().iloc[i])
                    
                    if is_buy:
                        # TP: Revert to Mean
                        if curr_close >= sma:
                            outcome_pnl = (curr_close - entry_price) - COSTS
                            break
                        # SL: Wide volatility stop (Mean Reversion needs breathing room)
                        # 3.0x Volatility Stop
                        stop_level = entry_price - (entry_price * vol * 3.0)
                        if curr_close <= stop_level:
                            outcome_pnl = (stop_level - entry_price) - COSTS
                            break
                    else: # Sell
                        # TP: Revert to Mean
                        if curr_close <= sma:
                            outcome_pnl = (entry_price - curr_close) - COSTS
                            break
                        # SL
                        stop_level = entry_price + (entry_price * vol * 3.0)
                        if curr_close >= stop_level:
                            outcome_pnl = (entry_price - stop_level) - COSTS
                            break
                
                # Time Exit (Failed to revert)
                if outcome_pnl == 0: 
                    exit_p = float(close.iloc[idx+24])
                    if is_buy: outcome_pnl = (exit_p - entry_price) - COSTS
                    else: outcome_pnl = (entry_price - exit_p) - COSTS
                
                # Risk Management (Fixed Fractional)
                # Risk 0.5% of Equity per trade
                risk_amt = equity * 0.005
                # Position Size = Risk / Stop Distance
                stop_dist = entry_price * vol * 3.0
                if stop_dist > 0:
                    units = risk_amt / stop_dist
                    trade_result = outcome_pnl * units
                    trades.append(trade_result)
                    equity += trade_result

            # Metrics
            if not trades:
                self.stdout.write(f"{era_name:<15} | 0      | N/A    | N/A    | $0         | NO TRADES")
                continue
                
            wins = sum(1 for t in trades if t > 0)
            losses = sum(1 for t in trades if t <= 0)
            gross_win = sum(t for t in trades if t > 0)
            gross_loss = abs(sum(t for t in trades if t <= 0))
            
            pf = gross_win / gross_loss if gross_loss > 0 else 99.9
            win_rate = wins / len(trades)
            net_profit = sum(trades)
            
            status = "FAIL"
            if pf >= 1.2: 
                status = "PASS"
            elif pf >= 1.0:
                status = "WEAK"
            
            color = self.style.SUCCESS if status == "PASS" else self.style.ERROR
            self.stdout.write(color(f"{era_name:<15} | {len(trades):<6} | {win_rate:.1%}  | {pf:.2f}   | ${net_profit:<9.0f} | {status}"))

            