# apps/analytics/management/commands/validate_trinity.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from django.core.management.base import BaseCommand
from apps.analytics.models.train import load_training_data, create_triple_barrier_target
from apps.market_data.models import Asset
from apps.analytics.features.pipeline import FeaturePipeline
from apps.market_data.services import OHLCVLoader
from datetime import datetime, timezone
from ta.trend import ADXIndicator, EMAIndicator
from ta.momentum import RSIIndicator

class Command(BaseCommand):
    help = "The Trinity Test: Trend Pullback Strategy Validation"

    def handle(self, *args, **options):
        # We test the big three: Euro, Yen, Gold
        assets = ['EURUSD', 'USDJPY', 'XAUUSD'] 
        timeframe = 'H1'
        
        self.stdout.write(self.style.WARNING("--- STARTING TRINITY TEST (STRATEGY: TREND PULLBACK) ---"))
        self.stdout.write("-" * 90)
        self.stdout.write(f"{'Asset':<8} | {'Base Win%':<10} | {'ML Win%':<10} | {'Trades':<8} | {'Status'}")
        self.stdout.write("-" * 90)

        for symbol in assets:
            self.run_asset_test(symbol, timeframe)

    def run_asset_test(self, symbol, tf):
        try:
            # 1. Load Data
            try:
                asset = Asset.objects.get(symbol=symbol)
            except Asset.DoesNotExist:
                self.stdout.write(f"{symbol:<8} | N/A        | N/A        | 0        | MISSING ASSET")
                return

            loader = OHLCVLoader()
            # 2024 Analysis Window
            start_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
            end_dt = datetime.now(timezone.utc)
            
            prices_df = loader.load_dataframe(asset, tf, start_dt, end_dt)
            
            if prices_df.empty or len(prices_df) < 500:
                self.stdout.write(f"{symbol:<8} | N/A        | N/A        | {len(prices_df):<8} | NO DATA")
                return

            # 2. Build Features (Regime Aware Pipeline)
            # This ensures we have vol_std and other ML features ready
            X = FeaturePipeline.build_feature_dataframe(symbol, prices_df)
            
            # --- 3. Signal Generation (Vectorized Logic for Pullback) ---
            # Replicating trend_pullback_signal logic globally
            ema_200 = EMAIndicator(close=prices_df['close'], window=200).ema_indicator()
            rsi = RSIIndicator(close=prices_df['close'], window=14).rsi()
            adx = ADXIndicator(high=prices_df['high'], low=prices_df['low'], close=prices_df['close'], window=14).adx()
            
            # The Pullback Logic Vectorized
            # Trend (Price > EMA200) AND Pullback (RSI < 45) AND Trend Strength (ADX > 20)
            signals = (prices_df['close'] > ema_200) & (rsi < 45) & (adx > 20)
            
            if signals.sum() < 30:
                # If signal count is low, it might be too strict or market was bearish
                self.stdout.write(f"{symbol:<8} | N/A        | N/A        | {signals.sum():<8} | FEW SIGNALS")
                return

            # --- 4. Ground Truth (Triple Barrier) ---
            # Adjusted for Pullback: We need time to recover, so horizon is wider (24 bars)
            # Stop is tighter than breakout (1.5x Vol) to cut losers fast
            outcomes = create_triple_barrier_target(
                prices=prices_df['close'],
                volatility=X['vol_std'],
                time_horizon=24,    # Give it a day to bounce
                pt_multiplier=2.0,  # 2.0x Vol Target
                sl_multiplier=1.5,  # 1.5x Vol Stop (Tighter risk)
                min_ret=0.0005
            )

            # --- 5. ML Training (Q1-Q3) vs Testing (Q4) ---
            # Filter Data based on Signal presence (Meta-Labeling)
            X_meta = X.loc[signals].copy()
            y_meta = outcomes.loc[signals]
            
            # Time Split
            split_date = datetime(2024, 10, 1, tzinfo=timezone.utc)
            train_mask = X_meta.index < split_date
            test_mask = X_meta.index >= split_date
            
            X_train, y_train = X_meta[train_mask], y_meta[train_mask]
            X_test, y_test = X_meta[test_mask], y_meta[test_mask]
            
            if len(X_train) < 20 or len(X_test) < 5:
                self.stdout.write(f"{symbol:<8} | N/A        | N/A        | Low Split | SKIP Q4")
                return

            # Train LightGBM
            model = lgb.LGBMClassifier(
                n_estimators=500, 
                max_depth=3, 
                learning_rate=0.02, 
                is_unbalance=True, 
                n_jobs=1, 
                verbose=-1, 
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Test
            base_winrate = y_test.mean()
            preds = model.predict_proba(X_test)[:, 1]
            
            # Filter at 0.60 threshold
            final_trades_mask = preds > 0.60
            
            if final_trades_mask.sum() == 0:
                ml_winrate = 0.0
            else:
                ml_winrate = y_test[final_trades_mask].mean()
            
            trades_count = final_trades_mask.sum()
            
            # Status Check Logic
            # PASS: ML Winrate > 55% (Strong Edge)
            # WEAK PASS: ML Winrate > Base (Some Edge)
            if ml_winrate > 0.55 and trades_count >= 5:
                status = self.style.SUCCESS("PASS")
            elif ml_winrate > base_winrate:
                status = self.style.WARNING("WEAK PASS")
            else:
                status = self.style.ERROR("FAIL")
                
            self.stdout.write(f"{symbol:<8} | {base_winrate:.1%}     | {ml_winrate:.1%}     | {trades_count:<8} | {status}")

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"{symbol}: Error {str(e)}"))