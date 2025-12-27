# apps/analytics/tests/conftest.py
import pytest
import tempfile
import joblib
import os
from django.utils import timezone
from apps.market_data.models import Asset
from apps.mlops.models import ModelRegistry
from apps.trading_core.models import FeatureVector, OHLCV
from sklearn.dummy import DummyClassifier

@pytest.fixture
def asset():
    """Provides a default Asset instance."""
    asset_obj, _ = Asset.objects.get_or_create(symbol="TEST_EURUSD")
    return asset_obj

@pytest.fixture
def active_model(asset):
    """Creates a dummy model, saves it, and registers it as active."""
    model = DummyClassifier(strategy="constant", constant=1)
    model.fit([[0, 0]], [0])
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib", dir=".") as tmp:
        joblib.dump(model, tmp.name)
        model_path = os.path.relpath(tmp.name)

    registry_entry = ModelRegistry.objects.create(
        version="v0.1.0-test",
        model_path=model_path,
        model_hash="test_hash",
        feature_list=["atr", "rsi"],
        is_active=True
    )
    
    yield registry_entry
    
    # Teardown
    os.remove(model_path)