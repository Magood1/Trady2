# apps/analytics/management/commands/research_feature_survival.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from django.core.management.base import BaseCommand
from django.conf import settings
from apps.analytics.services import OHLCVLoader
from apps.market_data.models import Asset
from datetime import datetime, timezone

class Command(BaseCommand):
    help = "Q1 2024 Simulation: Feature Importance & Survival Test"

    def add_arguments(self, parser):
        parser.add_argument('--symbol', type=str, default='EURUSD')
        parser.add_argument('--timeframe', type=str, default='H1')

    def handle(self, *args, **options):
        symbol = options['symbol']
        tf = options['timeframe']
        
        self.stdout.write(self.style.WARNING(f"--- STARTING Q1 2024 FEATURE SURVIVAL TEST FOR {symbol} ---"))

        # 1. Load Q1 Data (The Lab)
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 3, 31, tzinfo=timezone.utc)
        
        try:
            asset = Asset.objects.get(symbol=symbol)
        except Asset.DoesNotExist:
            self.stdout.write(self.style.ERROR(f"Asset {symbol} not found."))
            return

        loader = OHLCVLoader()
        try:
            df = loader.load_dataframe(asset, tf, start_date, end_date)
        except ValueError:
            self.stdout.write(self.style.ERROR("No data found for Q1 2024! Please ingest historical data first."))
            return
        
        if df.empty:
            self.stdout.write(self.style.ERROR("No data found for Q1 2024!"))
            return

        self.stdout.write(f"Loaded {len(df)} candles from Q1 2024.")

        # 2. Advanced Feature Engineering (In-Memory for Research)
        df = self.calculate_advanced_features(df)
        
        # 3. Add Random Noise Features (The Control Group)
        np.random.seed(42)
        df['shadow_noise_1'] = np.random.normal(0, 1, len(df))
        df['shadow_noise_2'] = np.random.uniform(0, 100, len(df))

        # 4. Define Proxy Target (Short-term volatility adjusted return)
        future_return = df['close'].shift(-1) / df['close'] - 1
        atr = df['high'] - df['low'] # Simple range proxy
        y = (future_return.abs() > (atr * 0.5)).astype(int)
        
        features = df.drop(columns=['open', 'high', 'low', 'close', 'volume']).fillna(0)
        
        # 5. Train LightGBM for Feature Importance
        self.stdout.write("Training LightGBM Probe Model...")
        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            verbose=-1
        )
        model.fit(features, y)

        # 6. Analyze Results
        importance = pd.DataFrame({
            'Feature': features.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        self.stdout.write(self.style.SUCCESS("\n--- FEATURE SURVIVAL RESULTS ---"))
        self.stdout.write(f"{'Feature':<25} | {'Score':<10} | {'Status':<10}")
        self.stdout.write("-" * 50)
        
        noise_threshold = importance[importance['Feature'].str.contains('shadow_noise')]['Importance'].max()
        
        for _, row in importance.iterrows():
            feat = row['Feature']
            score = row['Importance']
            
            if "shadow_noise" in feat:
                status = "NOISE BASELINE"
                color_func = self.style.WARNING
            elif score > noise_threshold * 1.2:
                status = "SURVIVED (ALPHA)"
                color_func = self.style.SUCCESS
            else:
                status = "REJECTED (JUNK)"
                color_func = self.style.ERROR
                
            self.stdout.write(color_func(f"{feat:<25} | {score:<10} | {status:<10}"))

    def calculate_advanced_features(self, df):
        df = df.copy()
        # Parkinson Volatility
        df['vol_parkinson'] = np.sqrt(
            (1.0 / (4.0 * np.log(2.0))) * np.log(df['high'] / df['low'])**2
        )
        
        # Amihud Illiquidity Proxy
        abs_ret = df['close'].pct_change().abs()
        df['liq_amihud'] = abs_ret / (df['volume'] + 1.0)
        
        # Efficiency Ratio
        period = 10
        change = (df['close'] - df['close'].shift(period)).abs()
        path = (df['close'] - df['close'].shift(1)).abs().rolling(window=period).sum()
        df['mom_efficiency'] = change / path.replace(0, 1)
        
        # Standard Indicators (Legacy Control)
        df['vol_std'] = df['close'].rolling(20).std()
        
        return df
    