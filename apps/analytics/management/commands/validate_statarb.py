# apps/analytics/management/commands/validate_statarb.py
import pandas as pd
import numpy as np
from django.core.management.base import BaseCommand
from apps.market_data.services import ingest_ohlcv_data
from apps.market_data.models import OHLCV, Asset
from apps.common.enums import Timeframe
from apps.analytics.quant.pairs_trading import PairsTradingEngine
from datetime import datetime
import pytz

class Command(BaseCommand):
    help = "Pairs Trading Stress Test: Rolling OLS with Decoupling Protection"

    def handle(self, *args, **options):
        # Pairs: Gold (Y) vs Silver (X)
        asset_y_sym = 'XAUUSD'
        asset_x_sym = 'XAGUSD'
        tf_str = 'H1'
        
        # Eras
        eras = {
            "Covid_Shock":  ("2020-01-01", "2021-12-31"),
            "Geo_Conflict": ("2022-01-01", "2024-12-18"),
        }

        self.stdout.write(self.style.WARNING(f"--- STAT-ARB TEST: {asset_y_sym} vs {asset_x_sym} ---"))
        self.stdout.write("Strategy: Rolling OLS | Z-Score Entry: 2.0 | Stop: 4.0 (Decoupling)")
        self.stdout.write("-" * 100)
        self.stdout.write(f"{'Era':<15} | {'Trades':<6} | {'Win%':<6} | {'Sharpe':<6} | {'Profit':<10} | {'MaxDD':<8} | {'Status'}")
        self.stdout.write("-" * 100)

        # Ensure Assets Exist
        try:
            asset_y = Asset.objects.get(symbol=asset_y_sym)
            asset_x = Asset.objects.get(symbol=asset_x_sym)
        except Asset.DoesNotExist:
            self.stdout.write(self.style.ERROR("Assets not found. Run ingestion first."))
            return

        for era_name, (start, end) in eras.items():
            start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
            end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
            
            # 1. Ingest (Best Effort)
            try:
                ingest_ohlcv_data(asset_y_sym, Timeframe(tf_str), start_dt, end_dt)
                ingest_ohlcv_data(asset_x_sym, Timeframe(tf_str), start_dt, end_dt)
            except: pass

            # 2. Load Data
            q_y = OHLCV.objects.filter(asset=asset_y, timeframe=tf_str, timestamp__range=(start_dt, end_dt)).order_by('timestamp')
            q_x = OHLCV.objects.filter(asset=asset_x, timeframe=tf_str, timestamp__range=(start_dt, end_dt)).order_by('timestamp')
            
            df_y = pd.DataFrame.from_records(q_y.values('timestamp', 'close'))
            df_x = pd.DataFrame.from_records(q_x.values('timestamp', 'close'))
            
            if len(df_y) < 500 or len(df_x) < 500:
                self.stdout.write(f"{era_name:<15} | NO DATA (Count mismatch)")
                continue
                
            # Align Time Series & Convert Types
            df_y['timestamp'] = pd.to_datetime(df_y['timestamp'], utc=True)
            df_y.set_index('timestamp', inplace=True)
            # FIX: Explicit float conversion from Decimal
            df_y['close'] = df_y['close'].astype(float)
            
            df_x['timestamp'] = pd.to_datetime(df_x['timestamp'], utc=True)
            df_x.set_index('timestamp', inplace=True)
            # FIX: Explicit float conversion from Decimal
            df_x['close'] = df_x['close'].astype(float)
            
            # 3. Math Engine
            data = PairsTradingEngine.get_aligned_data(df_y, df_x)
            
            if data.empty:
                self.stdout.write(f"{era_name:<15} | NO DATA (Alignment failed)")
                continue
                
            # Calculate Rolling OLS
            metrics = PairsTradingEngine.calculate_rolling_metrics(data, window=100)
            
            # 4. Simulation Loop
            ENTRY_Z = 2.0
            EXIT_Z = 0.0
            STOP_Z = 4.0 
            
            COST_Y = 0.40 # Spread+Comm per unit (Gold)
            COST_X = 0.03 # (Silver)
            
            equity = [10000.0]
            trades = []
            
            position = 0 
            entry_price_y = 0
            entry_price_x = 0
            entry_beta = 0
            
            for i in range(len(metrics)):
                row = metrics.iloc[i]
                z = row['z_score']
                price_y = row['y']
                price_x = row['x']
                beta = row['beta']
                
                # Check Exits first
                if position != 0:
                    stop_hit = (position == 1 and z < -STOP_Z) or (position == -1 and z > STOP_Z)
                    target_hit = (position == 1 and z >= EXIT_Z) or (position == -1 and z <= EXIT_Z)
                    
                    if stop_hit or target_hit:
                        # Position Sizing: 20% of Equity allocated
                        capital_alloc = equity[-1] * 0.20
                        
                        # Simplified PnL Calculation based on spread units
                        # PnL = (Exit_Spread - Entry_Spread) * Position_Dir * Multiplier
                        # But precise PnL requires calculating leg by leg:
                        
                        if position == 1: # Long Spread (Long Y, Short X)
                            pnl_y = (price_y - entry_price_y)
                            pnl_x = (entry_price_x - price_x) * beta
                        else: # Short Spread (Short Y, Long X)
                            pnl_y = (entry_price_y - price_y)
                            pnl_x = (price_x - entry_price_x) * beta
                            
                        gross_pnl_unit = pnl_y + pnl_x
                        
                        # Unit cost approx = Price Y + Beta * Price X
                        unit_cost = entry_price_y + (beta * entry_price_x)
                        num_units = capital_alloc / unit_cost if unit_cost > 0 else 0
                        
                        gross_pnl_total = gross_pnl_unit * num_units
                        t_cost = (COST_Y * num_units) + (COST_X * num_units * beta)
                        
                        net_pnl = gross_pnl_total - t_cost
                        
                        equity.append(equity[-1] + net_pnl)
                        trades.append(net_pnl)
                        
                        position = 0
                        continue

                # Check Entries
                if position == 0:
                    # Long Spread (Z < -2)
                    if z < -ENTRY_Z and z > -STOP_Z:
                        position = 1
                        entry_price_y = price_y
                        entry_price_x = price_x
                        entry_beta = beta
                    
                    # Short Spread (Z > 2)
                    elif z > ENTRY_Z and z < STOP_Z:
                        position = -1
                        entry_price_y = price_y
                        entry_price_x = price_x
                        entry_beta = beta

            # 5. Metrics Calculation
            if not trades:
                self.stdout.write(f"{era_name:<15} | 0      | N/A    | N/A    | $0         | 0%       | FLAT")
                continue

            trades_arr = np.array(trades)
            wins = (trades_arr > 0).sum()
            total = len(trades)
            win_rate = wins / total
            net_profit = trades_arr.sum()
            
            avg = trades_arr.mean()
            std = trades_arr.std()
            sharpe = (avg / std) * np.sqrt(total) if std > 0 else 0
            
            eq = pd.Series(equity)
            dd = (eq - eq.cummax()) / eq.cummax()
            max_dd = dd.min()
            
            status = "FAIL"
            if net_profit > 0 and max_dd > -0.20: status = "PASS"
            if net_profit > 2000 and max_dd > -0.10: status = "STRONG"
            
            color = self.style.SUCCESS if status == "STRONG" else self.style.WARNING
            self.stdout.write(color(
                f"{era_name:<15} | {total:<6} | {win_rate:.1%}  | {sharpe:.2f}   | ${net_profit:<9.0f} | {max_dd:.1%}   | {status}"
            ))

            