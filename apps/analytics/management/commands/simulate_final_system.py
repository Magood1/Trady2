# apps/analytics/management/commands/simulate_final_system.py
import pandas as pd
import numpy as np
from django.core.management.base import BaseCommand
from apps.analytics.models.train import load_aligned_data
from apps.market_data.models import Asset
from apps.mlops.services import get_active_model
from datetime import datetime, timezone

class Command(BaseCommand):
    help = "Quarterly Stress Test: Validates PF >= 1.3 across Q3/Q4"

    def add_arguments(self, parser):
        parser.add_argument('--symbol', type=str, default='XAUUSD')
        parser.add_argument('--timeframe', type=str, default='H1')
        parser.add_argument('--capital', type=float, default=10000.0)
        parser.add_argument('--stress', action='store_true', help="Double Costs Mode")

    def handle(self, *args, **options):
        symbol = options['symbol']
        stress = options['stress']
        initial_capital = options['capital']
        
        # 1. Cost Setup (Realism)
        if stress:
            self.stdout.write(self.style.ERROR("!!! STRESS MODE: SPREAD x2, SLIPPAGE x2 !!!"))
            SPREAD = 0.60
            COMMISSION = 0.20
            SLIP_FACTOR = 0.20
        else:
            SPREAD = 0.30
            COMMISSION = 0.10
            SLIP_FACTOR = 0.10

        # 2. Load Active Model & Verify Integrity
        model_info = get_active_model()
        if not model_info: 
            self.stdout.write(self.style.ERROR("No active model found."))
            return
        model, registry = model_info
        
        self.stdout.write(f"Auditing Model: {registry.version}")
        self.stdout.write(f"Expected Features: {registry.feature_list}")

        # 3. Load Data via Unified Pipeline
        try:
            asset = Asset.objects.get(symbol=symbol)
            X, prices_df = load_aligned_data(asset, options['timeframe'])
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Data Load Error: {e}"))
            return
        
        # Data Integrity Check
        if list(X.columns) != registry.feature_list:
            self.stdout.write(self.style.ERROR("CRITICAL: Feature Mismatch! Pipeline vs Model."))
            self.stdout.write(f"Pipeline: {list(X.columns)}")
            self.stdout.write(f"Model:    {registry.feature_list}")
            return

        # Filter 2024
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        mask = X.index >= start_date
        X = X[mask]
        prices_df = prices_df[mask]

        # 4. Simulation Loop (Strict Logic)
        # Base Signal Logic (From Pipeline Features directly)
        signals = (X['dist_ema200'] > 0) & \
                  (X['rsi'] < 45) & \
                  (X['adx'] > 20) & \
                  (X['is_green'] == 1.0)
        
        indices = np.where(signals)[0]
        self.stdout.write(f"Base Signals in 2024: {len(indices)}")
        
        if len(indices) == 0:
            self.stdout.write("No signals found.")
            return

        # ML Inference
        preds = model.predict_proba(X)[:, 1]
        
        trades = []
        equity = [initial_capital]
        
        for idx in indices:
            if idx + 24 >= len(prices_df): continue
            
            # ML Filter (0.60 Threshold)
            if preds[idx] < 0.60: continue
            
            # Trade Setup
            entry_price = float(prices_df['open'].iloc[idx+1]) # Open of next bar
            vol = float(X['vol_std'].iloc[idx])
            
            # Dynamic Stops
            stop_dist = max(2.0, entry_price * vol * 1.5)
            sl_price = entry_price - stop_dist
            tp_price = entry_price + (stop_dist * 2.0)
            
            # Outcome Loop
            outcome_pnl = 0
            
            for i in range(idx+1, idx+25):
                bar_low = float(prices_df['low'].iloc[i])
                bar_high = float(prices_df['high'].iloc[i])
                
                if bar_low <= sl_price:
                    # Loss + Slippage
                    slip = vol * entry_price * SLIP_FACTOR
                    exit_p = sl_price - slip
                    outcome_pnl = (exit_p - entry_price) - (SPREAD + COMMISSION)
                    break
                elif bar_high >= tp_price:
                    # Win - Cost
                    outcome_pnl = (tp_price - entry_price) - (SPREAD + COMMISSION)
                    break
            
            # Time Exit
            if outcome_pnl == 0:
                exit_p = float(prices_df['close'].iloc[idx+24])
                outcome_pnl = (exit_p - entry_price) - (SPREAD + COMMISSION)
            
            # Position Sizing (0.5% Risk)
            risk_amt = equity[-1] * 0.005
            units = risk_amt / stop_dist
            pnl_usd = outcome_pnl * units
            
            equity.append(equity[-1] + pnl_usd)
            trades.append({'date': prices_df.index[idx], 'pnl': pnl_usd, 'balance': equity[-1]})

        # 5. Quarterly Breakdown (The Verdict)
        df_t = pd.DataFrame(trades)
        if df_t.empty:
            self.stdout.write(self.style.WARNING("No trades executed after ML filter."))
            return

        df_t['quarter'] = df_t['date'].dt.to_period('Q')
        
        self.stdout.write("\n--- QUARTERLY AUDIT (STRESS TEST) ---")
        self.stdout.write(f"{'Q':<6} | {'Trades':<6} | {'Win%':<6} | {'Net PF':<6} | {'Profit':<10} | {'Status'}")
        self.stdout.write("-" * 60)
        
        valid_quarters = 0
        for q in sorted(df_t['quarter'].unique()):
            q_data = df_t[df_t['quarter'] == q]
            wins = q_data[q_data['pnl'] > 0]['pnl'].sum()
            losses = abs(q_data[q_data['pnl'] <= 0]['pnl'].sum())
            pf = wins / losses if losses > 0 else 99.9
            win_rate = (q_data['pnl'] > 0).mean()
            
            status = "FAIL"
            # Strict criteria: PF >= 1.3
            if pf >= 1.3 and len(q_data) >= 5: 
                status = "PASS"
                valid_quarters += 1
            elif pf >= 1.0:
                status = "WEAK"
                
            color = self.style.SUCCESS if status == "PASS" else self.style.ERROR
            self.stdout.write(color(f"{str(q):<6} | {len(q_data):<6} | {win_rate:.0%}    | {pf:.2f}   | ${q_data['pnl'].sum():.0f}      | {status}"))

        if valid_quarters >= 2:
            self.stdout.write(self.style.SUCCESS("\n*** VERDICT: SYSTEM READY FOR PILOT (2+ Quarters Passed) ***"))
        else:
            self.stdout.write(self.style.ERROR("\n*** VERDICT: SYSTEM REJECTED (Unstable) ***"))