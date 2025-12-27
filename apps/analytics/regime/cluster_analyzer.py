# apps/analytics/regime/cluster_analyzer.py
import pandas as pd
import structlog
import numpy as np  # <-- **الإضافة الحاسمة هنا**
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger(__name__)

def get_market_regimes(features_df: pd.DataFrame, n_regimes: int = 4, seed: int = 42) -> pd.Series:
    """
    Identifies market regimes using K-means clustering on key volatility and momentum features.
    Labels are stabilized by sorting clusters based on volatility, making them more interpretable.
    """
    feature_keys = ['garman_klass_vol', 'log_ret_24h', 'mean_reversion_dist_20']
    
    available_features = [f for f in feature_keys if f in features_df.columns]
    if not available_features:
        logger.warning("None of the specified features for clustering were found. Returning an empty series.", missing_keys=list(set(feature_keys) - set(features_df.columns)))
        return pd.Series(dtype='int')

    regime_features = features_df[available_features].copy()
    
    regime_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    regime_features.dropna(inplace=True)
    
    if regime_features.empty:
        logger.warning("Feature set for clustering is empty after dropping NaNs.")
        return pd.Series(dtype='int')

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(regime_features)
    
    kmeans = KMeans(n_clusters=n_regimes, random_state=seed, n_init='auto')
    regimes = kmeans.fit_predict(scaled_features)
    
    regime_series = pd.Series(regimes, index=regime_features.index, name='market_regime')
    
    cluster_volatility = regime_features.groupby(regime_series)['garman_klass_vol'].mean().sort_values()
    volatility_map = {old_label: new_label for new_label, old_label in enumerate(cluster_volatility.index)}
    
    logger.info("Market regimes clustered and sorted by volatility.", n_regimes=n_regimes)
    return regime_series.map(volatility_map)
