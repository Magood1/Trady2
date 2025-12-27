# apps/analytics/tests/test_train_pipeline.py
import pytest
import pandas as pd
from django.utils import timezone
from apps.analytics.models.train import create_target, load_training_data
from apps.market_data.models import OHLCV, Asset
from apps.trading_core.models import FeatureVector

@pytest.mark.django_db
def test_create_target_logic():
    prices = pd.Series([100, 101, 100.5, 102, 101],
                       index=pd.to_datetime(['2025-01-01 01:00', '2025-01-01 02:00', '2025-01-01 03:00', '2025-01-01 04:00', '2025-01-01 05:00']))
    target = create_target(prices, horizon=1, threshold=0.009) # 0.9%
    
    expected = pd.Series([1, 0, 1, 0], index=prices.index[:4], dtype='int32')
    pd.testing.assert_series_equal(target.dropna().astype('int32'), expected)

@pytest.mark.django_db
def test_load_data_alignment(asset):
    ts1 = timezone.now().replace(microsecond=0)
    ts2 = ts1 + pd.Timedelta(hours=1)
    
    FeatureVector.objects.create(asset=asset, timestamp=ts1, features={"atr": 0.1})
    OHLCV.objects.create(asset=asset, timeframe="H1", timestamp=ts1, open=1, high=1, low=1, close=100, volume=1)
    FeatureVector.objects.create(asset=asset, timestamp=ts2, features={"atr": 0.2})
    
    X, prices = load_training_data(asset, "H1")
    
    assert len(X) == 1
    assert X.index[0].replace(microsecond=0) == ts1

@pytest.mark.django_db
def test_load_training_data_handles_feature_named_close(asset):
    """Ensures the overlap error is fixed."""
    ts = timezone.now().replace(microsecond=0)
    # Create FeatureVector that includes 'close' as a feature
    FeatureVector.objects.create(asset=asset, timestamp=ts, features={"close": 123.45, "atr": 0.1})
    OHLCV.objects.create(asset=asset, timeframe="H1", timestamp=ts, open=1, high=1, low=1, close=1.2345, volume=1)
    
    X, prices = load_training_data(asset, "H1")
    
    assert 'close' not in X.columns
    assert prices.iloc[0] == 1.2345
    assert len(X) == 1