# apps/analytics/management/commands/validate_trinity_strict.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from django.core.management.base import BaseCommand
# FIX: Updated import to use 'load_aligned_data'
from apps.analytics.models.train import load_aligned_data, create_triple_barrier_target
from apps.market_data.models import Asset
from datetime import datetime, timezone
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator

class Command(BaseCommand):
    help = "The Financial Inquisition: Strict Logic, Dynamic Costs, Period Stability"

    def add_arguments(self, parser):
        parser.add_argument('--symbol', type=str, default='XAUUSD')
        parser.add_argument('--timeframe', type=str, default='H1')

    def handle(self, *args, **options):
        # Asset Config: Costs in raw price units (approximate)
        # EURUSD: Spread ~0.00015, Comm ~0.00007
        # XAUUSD: Spread ~0.30, Comm ~0.10
        # USDJPY: Spread ~0.015, Comm ~0.007
        assets_config = {
            'EURUSD': {'spread': 0.00015, 'comm': 0.00007}, 
            'USDJPY': {'spread': 0.015,   'comm': 0.007},   
            'XAUUSD': {'spread': 0.30,    'comm': 0.10},    
        }
        
        target_symbol = options['symbol']
        target_tf = options['timeframe']
        
        self.stdout.write(self.style.ERROR("--- STRICT FINANCIAL VALIDATION PROTOCOL ---"))
        self.stdout.write(f"Criteria: PF>=1.3 | Sharpe>=1.0 | DD<=10% | Stable across P1/P2")
        self.stdout.write("-" * 120)
        self.stdout.write(f"{'Asset':<8} | {'Per':<3} | {'Trades':<6} | {'Win%':<6} | {'Net PF':<6} | {'Sharpe':<6} | {'MaxDD':<6} | {'Status'}")
        self.stdout.write("-" * 120)

        # Run only for the requested symbol if specified in arguments (to match your workflow)
        if target_symbol in assets_config:
             self.run_strict_test(target_symbol, target_tf, assets_config[target_symbol])
        else:
             # Default fallback if symbol not in map, assume Gold-like costs or fail
             self.stdout.write(self.style.WARNING(f"Symbol {target_symbol} not in config map. Using XAUUSD costs."))
             self.run_strict_test(target_symbol, target_tf, assets_config['XAUUSD'])

    def run_strict_test(self, symbol, tf, base_costs):
        try:
            asset = Asset.objects.get(symbol=symbol)
            # FIX: Use the new aligned loader
            X, prices_df = load_aligned_data(asset, tf)
            
            # --- 1. Signal Generation (Strict Entry) ---
            # Trend Pullback + Green Candle Confirmation
            ema_200 = EMAIndicator(close=prices_df['close'], window=200).ema_indicator()
            rsi = RSIIndicator(close=prices_df['close'], window=14).rsi()
            adx = ADXIndicator(high=prices_df['high'], low=prices_df['low'], close=prices_df['close'], window=14).adx()
            
            # Filter Logic
            signals = (prices_df['close'] > ema_200) & \
                      (rsi < 45) & \
                      (adx > 20) & \
                      (prices_df['close'] > prices_df['open'])
            
            if signals.sum() < 50: # Pre-filter for sample size (lowered threshold for higher TFs)
                self.stdout.write(f"{symbol:<8} | ALL | {signals.sum():<6} | N/A    | N/A    | N/A    | N/A    | REJECTED (Sample < 50)")
                return

            # --- 2. Ground Truth (Target) ---
            # Using TBM with 2.0 Reward / 1.5 Risk
            outcomes = create_triple_barrier_target(
                prices=prices_df['close'],
                volatility=X['vol_std'],
                time_horizon=24,
                pt_multiplier=2.0,
                sl_multiplier=1.5,
                min_ret=0.0005
            )

            # --- 3. ML Training/Testing Split ---
            X_meta = X.loc[signals].copy()
            y_meta = outcomes.loc[signals]
            prices_meta = prices_df.loc[signals]
            
            # Split Dates (Simulating 2024 progression)
            # Use data relative to the end of the dataset to be robust for different TFs
            end_date = X_meta.index.max()
            try:
                # Dynamically set split points based on available data range
                # P2: Last 3 months
                # P1: 3 months before that
                # Train: Everything before
                d2 = end_date - pd.Timedelta(days=90)
                d1 = d2 - pd.Timedelta(days=90)
                
                # Convert to timezone aware if needed to match index
                if X_meta.index.tz:
                    d1 = d1.replace(tzinfo=timezone.utc)
                    d2 = d2.replace(tzinfo=timezone.utc)
            except:
                # Fallback hardcoded if datetime arithmetic fails
                d1 = datetime(2024, 7, 1, tzinfo=timezone.utc)
                d2 = datetime(2024, 10, 1, tzinfo=timezone.utc)

            train_mask = X_meta.index < d1
            p1_mask = (X_meta.index >= d1) & (X_meta.index < d2)
            p2_mask = X_meta.index >= d2
            
            if train_mask.sum() < 50:
                 self.stdout.write(f"{symbol:<8} | ERR | Insufficient training data before split.")
                 return
            
            # Train Model Once
            model = lgb.LGBMClassifier(n_estimators=500, max_depth=3, learning_rate=0.02, is_unbalance=True, n_jobs=1, verbose=-1, random_state=42)
            model.fit(X_meta[train_mask], y_meta[train_mask])
            
            # --- 4. Strict Evaluation Loop ---
            periods = [('P1', p1_mask), ('P2', p2_mask)]
            statuses = []
            
            for p_name, mask in periods:
                if mask.sum() < 10: 
                    self.stdout.write(f"{symbol:<8} | {p_name:<3} | {mask.sum():<6} | N/A    | N/A    | N/A    | N/A    | REJECTED (Low Data)")
                    statuses.append('FAIL')
                    continue

                preds = model.predict_proba(X_meta[mask])[:, 1]
                
                # Threshold
                trades_mask = preds > 0.60
                
                if trades_mask.sum() == 0:
                    self.stdout.write(f"{symbol:<8} | {p_name:<3} | 0      | N/A    | N/A    | N/A    | N/A    | REJECTED (No Trades)")
                    statuses.append('FAIL')
                    continue
                
                # --- Financial Simulation ---
                p_prices = prices_meta.loc[mask][trades_mask]
                p_vol = X_meta.loc[mask, 'vol_std'][trades_mask]
                p_outcome = y_meta.loc[mask][trades_mask] # 1=Win, 0=Loss
                
                equity_curve = [10000.0]
                wins = 0
                losses = 0
                gross_profit = 0
                gross_loss = 0
                
                for i in range(len(p_prices)):
                    price = p_prices['close'].iloc[i]
                    vol = p_vol.iloc[i]
                    is_win = p_outcome.iloc[i] == 1
                    
                    # 1. Dynamic Slippage
                    slippage = price * vol * 0.1 
                    
                    # 2. Total Cost
                    total_cost = base_costs['spread'] + base_costs['comm'] + slippage
                    
                    # 3. PnL Calculation
                    r_dist = price * vol * 1.5
                    
                    if r_dist == 0: continue 
                    
                    if is_win:
                        # Net Profit = Target Distance - Cost
                        pnl = (r_dist * (2.0/1.5)) - total_cost
                        if pnl > 0: 
                            wins += 1
                            gross_profit += pnl
                        else:
                            losses += 1
                            gross_loss += abs(pnl)
                    else:
                        # Net Loss = Stop Distance + Cost
                        pnl = -(r_dist + total_cost)
                        losses += 1
                        gross_loss += abs(pnl)
                    
                    # Update Equity (Fixed 1% Risk Sizing)
                    if equity_curve[-1] > 0:
                        pos_size = (equity_curve[-1] * 0.01) / r_dist
                        trade_pnl_usd = pnl * pos_size
                        equity_curve.append(equity_curve[-1] + trade_pnl_usd)
                    else:
                        equity_curve.append(0)

                # --- Metrics Calculation ---
                total_trades = wins + losses
                win_rate = wins / total_trades if total_trades > 0 else 0
                pf = gross_profit / gross_loss if gross_loss > 0 else 0
                
                # Sharpe
                eq_series = pd.Series(equity_curve)
                returns = eq_series.pct_change().dropna()
                if returns.std() > 0:
                    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) # Annualized
                else:
                    sharpe = 0
                
                # Max DD
                cum_max = eq_series.cummax()
                drawdown = (eq_series - cum_max) / cum_max
                max_dd = drawdown.min()
                
                # --- Strict Judgment ---
                status = "REJECTED"
                color = self.style.ERROR
                
                if total_trades < 10: 
                     status = "LOW SAMPLE"
                elif pf >= 1.3 and sharpe >= 1.0 and max_dd >= -0.10:
                    status = "PASS"
                    color = self.style.SUCCESS
                elif pf >= 1.1:
                    status = "BORDERLINE"
                    color = self.style.WARNING
                
                self.stdout.write(color(f"{symbol:<8} | {p_name:<3} | {total_trades:<6} | {win_rate:.1%}   | {pf:.2f}   | {sharpe:.2f}   | {max_dd:.1%}  | {status}"))
                statuses.append(status)

            # Final Verdict
            if statuses and all(s == 'PASS' for s in statuses):
                self.stdout.write(self.style.SUCCESS(f"*** {symbol} IS READY FOR DEPLOYMENT ***"))
            else:
                self.stdout.write(self.style.ERROR(f"*** {symbol} FAILED STABILITY CHECK ***"))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"{symbol}: Critical Error {str(e)}"))
            