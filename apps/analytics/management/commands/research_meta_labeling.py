# apps/analytics/management/commands/research_meta_labeling.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from django.core.management.base import BaseCommand
from django.conf import settings
from apps.analytics.models.train import load_training_data, create_triple_barrier_target
from apps.market_data.models import Asset
from datetime import datetime, timezone
from sklearn.metrics import precision_score, recall_score, accuracy_score

class Command(BaseCommand):
    help = "Meta-Labeling Research: Can ML fix a low-winrate strategy?"

    def add_arguments(self, parser):
        parser.add_argument('--symbol', type=str, default='EURUSD')
        parser.add_argument('--timeframe', type=str, default='H1')

    def handle(self, *args, **options):
        symbol = options['symbol']
        tf = options['timeframe']
        
        self.stdout.write(self.style.WARNING(f"--- STARTING META-LABELING RESEARCH FOR {symbol} ({tf}) ---"))

        # 1. Load Data
        try:
            asset = Asset.objects.get(symbol=symbol)
        except Asset.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"Asset '{symbol}' does not exist in DB. Please run ingestion task."))
            return

        # Using the standard loader which gives us vol_std and survivor features
        try:
            X, prices_df = load_training_data(asset, tf)
        except ValueError as e:
            self.stdout.write(self.style.ERROR(f"Data Error: {str(e)}"))
            self.stdout.write(self.style.NOTICE("HINT: If you just ran ingestion, wait 1-2 minutes for Celery to finish writing to DB."))
            return
        
        # Filter Data (Adjusted for typical available data in your sample)
        # Using a wider range to ensure we catch whatever data is in the DB
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        
        mask = (X.index >= start_date)
        
        if mask.sum() == 0:
             # Fallback: Use all data if 2024 specific filter yields nothing
             self.stdout.write(self.style.WARNING("No 2024 data found. Using ALL available data for research."))
             X_research = X.copy()
             prices_research = prices_df.copy()
        else:
             X_research = X[mask].copy()
             prices_research = prices_df[mask].copy()
        
        if len(X_research) < 100:
            self.stdout.write(self.style.ERROR(f"Not enough data rows ({len(X_research)}). Need at least 100."))
            return

        self.stdout.write(f"Analyzing {len(X_research)} candles...")

        # 2. Define The 'Dumb' Base Strategy (The Signal Generator)
        # Strategy: Buy when Price closes above Upper Bollinger Band
        bb = BollingerBands(close=prices_research['close'], window=20, window_dev=2)
        rsi_ind = RSIIndicator(close=prices_research['close'], window=14)
        
        rsi = rsi_ind.rsi()
        bb_upper = bb.bollinger_hband()
        bb_width = bb.bollinger_wband()
        
        # Signal Logic: Breakout + Not Overbought (>75)
        # We relaxed RSI slightly to allow strong momentum
        long_signals = (prices_research['close'] > bb_upper) & (rsi < 75)
        
        if long_signals.sum() == 0:
            self.stdout.write(self.style.ERROR("Base strategy generated 0 signals on this data."))
            return

        self.stdout.write(f"Base Strategy Generated {long_signals.sum()} signals.")

        # 3. Create Ground Truth (Target) using TBM
        # ADAPTATION: Breakout strategies need wider stops. 
        # Changing sl_multiplier from 1.0 to 2.0 to fix the "1.43% Win Rate" issue.
        outcomes = create_triple_barrier_target(
            prices=prices_research['close'],
            volatility=X_research['vol_std'],
            time_horizon=12,
            pt_multiplier=2.0,  # Aim for big moves
            sl_multiplier=2.0,  # Give it room to breathe (was 1.0)
            min_ret=0.0005
        )

        # 4. Prepare Meta-Data for ML
        # Filter only rows where Signal == True
        X_meta = X_research.loc[long_signals].copy()
        y_meta = outcomes.loc[long_signals]
        
        # Add Context Features specifically for the Meta Model
        # "Why does this specific breakout fail?"
        X_meta['rsi_at_entry'] = rsi.loc[long_signals]
        X_meta['bb_width_at_entry'] = bb_width.loc[long_signals]
        X_meta['vol_at_entry'] = X_research['vol_std'].loc[long_signals]
        
        # Fill NaNs
        X_meta.fillna(0, inplace=True)
        
        # Split Train/Test (Chronological)
        train_size = int(len(X_meta) * 0.7)
        if train_size < 10:
            self.stdout.write(self.style.ERROR("Not enough signals to split train/test."))
            return

        X_train, X_test = X_meta.iloc[:train_size], X_meta.iloc[train_size:]
        y_train, y_test = y_meta.iloc[:train_size], y_meta.iloc[train_size:]
        
        base_win_rate = y_test.mean()
        self.stdout.write(f"Base Strategy Win Rate (Test Set): {base_win_rate:.2%}")

        # 5. Train Meta-Model
        # Setting n_jobs=1 to fix Windows multiprocessing issue
        model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.02,
            is_unbalance=True, 
            random_state=42,
            n_jobs=1,  # FIX for Windows
            verbose=-1
        )
        model.fit(X_train, y_train)

        # 6. Evaluate The "Gatekeeper"
        preds_proba = model.predict_proba(X_test)[:, 1]
        
        self.stdout.write("\n--- META-MODEL PERFORMANCE (The Filter Effect) ---")
        self.stdout.write(f"{'Threshold':<10} | {'Trades':<8} | {'New WinRate':<12} | {'Uplift':<10}")
        self.stdout.write("-" * 50)
        
        for thresh in [0.5, 0.55, 0.6, 0.65, 0.7]:
            filtered_trades_mask = (preds_proba >= thresh)
            trades_count = filtered_trades_mask.sum()
            
            if trades_count == 0:
                self.stdout.write(f"{thresh:<10} | 0        | N/A          | N/A")
                continue
                
            new_win_rate = y_test[filtered_trades_mask].mean()
            uplift = new_win_rate - base_win_rate
            
            # Formatting output
            win_rate_str = f"{new_win_rate:.2%}"
            uplift_str = f"{uplift:+.2%}"
            
            if uplift > 0.05: # >5% improvement
                color = self.style.SUCCESS
            elif uplift < 0:
                color = self.style.ERROR
            else:
                color = self.style.WARNING
            
            self.stdout.write(color(
                f"{thresh:<10} | {trades_count:<8} | {win_rate_str:<12} | {uplift_str:<10}"
            ))

        self.stdout.write(self.style.NOTICE("\nINTERPRETATION:"))
        self.stdout.write("1. If 'Base Strategy Win Rate' is still < 10%, verify OHLCV data quality or widen Triple Barrier stops.")
        self.stdout.write("2. Look for a threshold where WinRate jumps significantly (e.g., from 30% to 50%+).")