# apps/analytics/management/commands/research_tbm_stability.py
import pandas as pd
import lightgbm as lgb
from django.core.management.base import BaseCommand
from apps.analytics.models.train import load_training_data, create_triple_barrier_target
from apps.market_data.models import Asset
from datetime import datetime, timezone
from sklearn.metrics import precision_score, roc_auc_score

class Command(BaseCommand):
    help = "Q3 2024: Stability Test with Triple Barrier Method (TBM)"

    def add_arguments(self, parser):
        parser.add_argument('--symbol', type=str, default='EURUSD')
        parser.add_argument('--timeframe', type=str, default='H1')

    def handle(self, *args, **options):
        symbol = options['symbol']
        tf = options['timeframe']
        
        self.stdout.write(self.style.WARNING(f"--- TESTING TRIPLE BARRIER STABILITY FOR {symbol} ---"))

        try:
            asset = Asset.objects.get(symbol=symbol)
        except Asset.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"Asset {symbol} not found."))
            return
        
        # 1. Load Data (Using the new loader with 'vol_std')
        self.stdout.write("Loading data and computing survivor features...")
        X, prices_df = load_training_data(asset, tf)
        
        if X.empty:
            self.stdout.write(self.style.ERROR("No data found. Please ingest historical data."))
            return

        # 2. Create TBM Target (The Solution)
        # Using 1.5x Vol for Profit, 1.0x Vol for Stop
        self.stdout.write("Generating Triple Barrier Targets...")
        y = create_triple_barrier_target(
            prices=prices_df['close'],
            volatility=X['vol_std'],
            time_horizon=12,  # 12 hours horizon
            pt_multiplier=1.5,
            sl_multiplier=1.0
        )
        
        # Filter for Q2 (Train) and Q3 (Test)
        # Assuming index is tz-aware UTC
        q2_start = datetime(2024, 4, 1, tzinfo=timezone.utc)
        q2_end = datetime(2024, 6, 30, tzinfo=timezone.utc)
        q3_start = datetime(2024, 7, 1, tzinfo=timezone.utc)
        q3_end = datetime(2024, 7, 31, tzinfo=timezone.utc)
        
        train_mask = (X.index >= q2_start) & (X.index <= q2_end)
        test_mask = (X.index >= q3_start) & (X.index <= q3_end)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        if len(X_train) == 0 or len(X_test) == 0:
             self.stdout.write(self.style.ERROR("Insufficient data overlap with Q2/Q3 2024 dates."))
             return
        
        # Check Class Balance
        pos_ratio = y_train.mean()
        self.stdout.write(f"Training Class Balance (Win Rate): {pos_ratio:.2%}")
        
        if pos_ratio < 0.05:
            self.stdout.write(self.style.ERROR("Warning: Targets are too sparse! Need to adjust TBM parameters (e.g., lower multiplier)."))

        # 3. Train Model
        self.stdout.write("Training Model on TBM Target...")
        model = lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
        model.fit(X_train, y_train)

        # 4. Daily Stability Check
        self.stdout.write("\n--- TBM STABILITY ANALYSIS ---")
        self.stdout.write(f"{'Day':<5} | {'Precision':<10} | {'AUC':<10} | {'Trades':<8}")
        self.stdout.write("-" * 50)
        
        test_df = X_test.copy()
        test_df['target'] = y_test
        test_df['day'] = test_df.index.date
        unique_days = sorted(test_df['day'].unique())
        
        for i, day in enumerate(unique_days):
            daily = test_df[test_df['day'] == day]
            if len(daily) < 10: continue
            
            # Predict
            preds_proba = model.predict_proba(daily[X.columns])[:, 1]
            
            # Using a probability threshold to filter high-quality trades
            threshold = 0.55
            preds_class = (preds_proba > threshold).astype(int)
            
            try:
                # Handle cases where only one class is present in target or prediction
                if len(daily['target'].unique()) > 1:
                    auc = roc_auc_score(daily['target'], preds_proba)
                else:
                    auc = 0.5
                
                prec = precision_score(daily['target'], preds_class, zero_division=0)
                trades = preds_class.sum()
            except Exception as e:
                auc = 0.0
                prec = 0.0
                trades = 0
            
            # Color coding for "Good" days
            if prec > 0.55: 
                color = self.style.SUCCESS
            elif prec < 0.40 and trades > 0: 
                color = self.style.ERROR
            else: 
                color = self.style.WARNING
            
            self.stdout.write(color(f"Day {i+1:<3} | {prec:.4f}     | {auc:.4f}     | {trades}"))

        self.stdout.write(self.style.NOTICE("\nINTERPRETATION:"))
        self.stdout.write("- Precision > 0.55 is the goal. It means >55% of trades taken were profitable.")
        self.stdout.write("- AUC measures ranking ability (0.5 is random).")
        self.stdout.write("- If stability is achieved, integrate TBM into production pipeline.")

        