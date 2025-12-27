# apps/analytics/management/commands/research_tbm_tuning.py
import pandas as pd
import numpy as np
from itertools import product
from django.core.management.base import BaseCommand
from apps.analytics.models.train import load_training_data, create_triple_barrier_target
from apps.market_data.models import Asset
from datetime import datetime, timezone

class Command(BaseCommand):
    help = "Find the Golden Ratio: Tuning Triple Barrier Parameters"

    def add_arguments(self, parser):
        parser.add_argument('--symbol', type=str, default='EURUSD')
        parser.add_argument('--timeframe', type=str, default='H1')

    def handle(self, *args, **options):
        symbol = options['symbol']
        tf = options['timeframe']
        
        self.stdout.write(self.style.WARNING(f"--- STARTING TBM CALIBRATION FOR {symbol} ---"))

        # 1. Load Data (Q1 & Q2 2024 as calibration set)
        asset = Asset.objects.get(symbol=symbol)
        X, prices_df = load_training_data(asset, tf)
        
        # Filter for H1 2024
        mask = (X.index >= datetime(2024, 1, 1, tzinfo=timezone.utc)) & \
               (X.index <= datetime(2024, 6, 30, tzinfo=timezone.utc))
        prices = prices_df.loc[mask, 'close']
        volatility = X.loc[mask, 'vol_std'] # The Survivor Feature
        
        if len(prices) == 0:
            self.stdout.write(self.style.ERROR("No data for calibration period."))
            return

        # 2. Define Parameter Grid
        # We look for a setup that naturally provides opportunities
        pt_multipliers = [1.0, 1.5, 2.0, 2.5, 3.0] # Profit targets relative to Vol
        sl_multipliers = [0.5, 1.0, 1.5, 2.0]      # Stop losses relative to Vol
        horizons = [6, 12, 24]                     # Time limits (bars)

        results = []

        self.stdout.write(f"Testing {len(pt_multipliers)*len(sl_multipliers)*len(horizons)} combinations...")
        self.stdout.write("-" * 60)
        self.stdout.write(f"{'PT':<5} | {'SL':<5} | {'Hrs':<4} | {'WinRate%':<10} | {'Trades':<8} | {'Score'}")
        self.stdout.write("-" * 60)

        for pt, sl, hrs in product(pt_multipliers, sl_multipliers, horizons):
            # Skip unrealistic ratios (Risk > Reward usually bad for ML unless high winrate)
            if sl > pt: continue 

            targets = create_triple_barrier_target(
                prices=prices,
                volatility=volatility,
                time_horizon=hrs,
                pt_multiplier=pt,
                sl_multiplier=sl,
                min_ret=0.0005 # Covers basic spread cost
            )
            
            num_trades = targets.sum()
            total_samples = len(targets)
            win_rate = num_trades / total_samples if total_samples > 0 else 0
            
            # Custom Score: We want balance. 
            # Too few trades = starvation. Too many = noise.
            # We prefer High PT/SL ratio (Reward) combined with decent WinRate.
            
            # Simple heuristic score: WinRate * (PT/SL)
            # If WinRate is < 5%, score is penalized heavily (Silent Death zone)
            if win_rate < 0.05:
                score = 0
            else:
                risk_reward = pt / sl
                score = win_rate * risk_reward * 100

            results.append({
                'pt': pt, 'sl': sl, 'hrs': hrs,
                'win_rate': win_rate, 'trades': num_trades, 'score': score
            })
            
            # Highlight promising candidates
            if win_rate > 0.15 and score > 20:
                self.stdout.write(self.style.SUCCESS(
                    f"{pt:<5.1f} | {sl:<5.1f} | {hrs:<4} | {win_rate:.2%}     | {num_trades:<8} | {score:.2f}"
                ))
            elif win_rate > 0.05:
                 self.stdout.write(
                    f"{pt:<5.1f} | {sl:<5.1f} | {hrs:<4} | {win_rate:.2%}     | {num_trades:<8} | {score:.2f}"
                )

        # 3. Recommendation
        if not results: return
        
        best_cfg = max(results, key=lambda x: x['score'])
        self.stdout.write(self.style.WARNING("\n--- WINNER CONFIGURATION ---"))
        self.stdout.write(str(best_cfg))
        self.stdout.write("Use these parameters in your production pipeline.")

        