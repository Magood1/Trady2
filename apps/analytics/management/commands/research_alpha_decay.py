# apps/analytics/management/commands/research_alpha_decay.py
import pandas as pd
import lightgbm as lgb
from django.core.management.base import BaseCommand
from apps.analytics.services import OHLCVLoader
from apps.market_data.models import Asset
from datetime import datetime, timedelta, timezone
from sklearn.metrics import roc_auc_score

class Command(BaseCommand):
    help = "Q3 2024 Simulation: Alpha Decay Measurement"

    def add_arguments(self, parser):
        parser.add_argument('--symbol', type=str, default='EURUSD')
        parser.add_argument('--timeframe', type=str, default='H1')

    def handle(self, *args, **options):
        symbol = options['symbol']
        tf = options['timeframe']
        
        self.stdout.write(self.style.WARNING(f"--- STARTING Q3 2024 ALPHA DECAY TEST FOR {symbol} ---"))

        # 1. Define Timeline
        train_start = datetime(2024, 4, 1, tzinfo=timezone.utc)
        train_end = datetime(2024, 6, 30, tzinfo=timezone.utc)
        test_start = datetime(2024, 7, 1, tzinfo=timezone.utc)
        test_end = datetime(2024, 7, 31, tzinfo=timezone.utc)

        try:
            asset = Asset.objects.get(symbol=symbol)
        except Asset.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"Asset {symbol} not found."))
            return

        loader = OHLCVLoader()
        
        try:
            df_train = loader.load_dataframe(asset, tf, train_start, train_end)
            df_test = loader.load_dataframe(asset, tf, test_start, test_end)
        except ValueError:
             self.stdout.write(self.style.ERROR("Missing Q2/Q3 2024 data. Cannot proceed."))
             return
        
        if df_train.empty or df_test.empty:
            self.stdout.write(self.style.ERROR("Missing Q2/Q3 2024 data (empty df)."))
            return

        # 2. Prep Data
        for df in [df_train, df_test]:
            df['ret'] = df['close'].pct_change()
            df['vol'] = df['ret'].rolling(20).std()
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            df.fillna(0, inplace=True)

        feature_cols = ['vol', 'volume']
        
        # 3. Train "Static" Model
        self.stdout.write("Training Static Model on Q2 Data...")
        model = lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
        model.fit(df_train[feature_cols], df_train['target'])

        # 4. Measure Decay Day by Day
        self.stdout.write("\n--- DECAY ANALYSIS (Performance vs. Age of Model) ---")
        self.stdout.write(f"{'Day':<5} | {'AUC Score':<10} | {'Status':<15}")
        self.stdout.write("-" * 40)

        df_test['day'] = df_test.index.date
        unique_days = sorted(df_test['day'].unique())
        
        baseline_auc = 0.5
        
        for i, day in enumerate(unique_days):
            daily_data = df_test[df_test['day'] == day]
            if len(daily_data) < 10: continue
            
            preds = model.predict_proba(daily_data[feature_cols])[:, 1]
            try:
                if len(daily_data['target'].unique()) > 1:
                    auc = roc_auc_score(daily_data['target'], preds)
                else:
                    auc = 0.5 # Default if only one class exists
            except ValueError:
                auc = 0.5
            
            if i == 0: baseline_auc = auc
            
            drop_pct = (baseline_auc - auc) / (baseline_auc if baseline_auc > 0 else 1)
            
            status = "HEALTHY"
            color = self.style.SUCCESS
            
            if auc < 0.51:
                status = "DEAD (Random)"
                color = self.style.ERROR
            elif drop_pct > 0.10: 
                status = "DECAYING"
                color = self.style.WARNING
            
            self.stdout.write(color(f"Day {i+1:<3} | {auc:.4f}     | {status}"))

        self.stdout.write(self.style.NOTICE("\nINTERPRETATION: Check specifically when status turns DECAYING consistently."))

        