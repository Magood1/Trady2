# apps/analytics/models/train.py
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Tuple
from apps.market_data.models import Asset
from apps.mlops.services import save_model
from apps.analytics.features.pipeline import FeaturePipeline
from apps.analytics.services import OHLCVLoader

def create_triple_barrier_target(
    prices: pd.Series, volatility: pd.Series, time_horizon: int = 24,
    pt_multiplier: float = 2.0, sl_multiplier: float = 1.5, min_ret: float = 0.0005
) -> pd.Series:
    """Vectorized TBM Target Generation (Optimized)"""
    upper = prices * (1 + volatility * pt_multiplier)
    lower = prices * (1 - volatility * sl_multiplier)
    
    T = len(prices)
    price_vals = prices.values
    first_touch_up = np.full(T, np.inf)
    first_touch_down = np.full(T, np.inf)
    
    for step in range(1, time_horizon + 1):
        future_p = np.full(T, np.nan)
        future_p[:-step] = price_vals[step:]
        
        # Check targets
        hit_up = (future_p >= upper.values) & (first_touch_up == np.inf)
        first_touch_up[hit_up] = step
        
        # Check stops
        hit_down = (future_p <= lower.values) & (first_touch_down == np.inf)
        first_touch_down[hit_down] = step
        
    is_profit = (first_touch_up != np.inf) & (first_touch_up < first_touch_down)
    is_worth = (volatility * pt_multiplier) > min_ret
    return pd.Series((is_profit & is_worth).astype(int), index=prices.index)

def load_aligned_data(asset: Asset, timeframe: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    loader = OHLCVLoader()
    # Load ALL available data to ensure indicators (EMA/RSI) warm up correctly
    prices_df = loader.load_dataframe(asset, timeframe, pd.Timestamp.min.tz_localize('UTC'), pd.Timestamp.max.tz_localize('UTC'))
    
    if prices_df.empty:
        raise ValueError(f"No OHLCV data for {asset.symbol}")

    # FORCE usage of FeaturePipeline
    features_df = FeaturePipeline.build_feature_dataframe(asset.symbol, prices_df)
    
    # Align indices (Drop rows where features are NaN due to warmup)
    common_index = prices_df.index.intersection(features_df.index)
    # Ensure critical volatility feature is valid
    valid_mask = features_df.loc[common_index, 'vol_std'] > 0 
    
    return features_df.loc[common_index][valid_mask], prices_df.loc[common_index][valid_mask]

def run_training_pipeline(config: Dict, seed: int):
    print(f"--- STRICT TRAINING PROTOCOL FOR {config['training']['asset_symbol']} ---")
    
    try:
        asset = Asset.objects.get(symbol=config['training']['asset_symbol'])
    except Exception as e:
        print(f"Error finding asset: {e}")
        return

    # 1. Load Data (Unified)
    try:
        X, prices = load_aligned_data(asset, config['training']['timeframe'])
    except ValueError as e:
        print(e)
        return
        
    print(f"Loaded Data: {len(X)} rows. Features: {list(X.columns)}")

    # 2. Generate Target
    # Using pipeline feature 'vol_std' directly
    y = create_triple_barrier_target(
        prices['close'], X['vol_std'],
        time_horizon=24, pt_multiplier=2.0, sl_multiplier=1.5
    )

    # 3. Base Strategy Filter (Must match Strategy Logic exactly)
    # Logic: Close > EMA200 & RSI < 45 & ADX > 20 & Green Candle
    # All required inputs are now in X from the Pipeline
    
    signals = (X['dist_ema200'] > 0) & \
              (X['rsi'] < 45) & \
              (X['adx'] > 20) & \
              (X['is_green'] == 1.0)
    
    X_meta = X.loc[signals].copy()
    y_meta = y.loc[signals].copy()
    
    print(f"Qualified Signals for Training: {len(X_meta)}")
    if len(X_meta) < 50:
        print("WARNING: Insufficient signals. Training might be unstable.")

    # 4. Split & Train
    split_date = pd.to_datetime(config['data_split']['split_date'], utc=True)
    
    X_train = X_meta[X_meta.index < split_date]
    y_train = y_meta[y_meta.index < split_date]
    X_test = X_meta[X_meta.index >= split_date]
    y_test = y_meta[y_meta.index >= split_date]
    
    print(f"Training Set: {len(X_train)} | Test Set: {len(X_test)}")
    
    # 5. LightGBM Hygiene
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=3,            
        num_leaves=8,           
        min_child_samples=10,   
        min_split_gain=0.01,    
        is_unbalance=True,
        random_state=seed,
        n_jobs=1,               
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    # 6. Save with Explicit Feature List
    save_model(model, config, {"best_score": float(model.best_score_['valid_0']['auc'])}, X_train.columns.tolist(), model_name_suffix="strict_v1")
    print("Model Saved. Feature Hash Locked.")