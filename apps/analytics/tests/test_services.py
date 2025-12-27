import datetime

import pandas as pd
import pytest
import numpy as np 
from django.conf import settings
from django.utils import timezone
import fbm 

from apps.analytics.services import (
    FeatureEngineer,
    OHLCVLoader,
    RegimeClassifier,
)
from apps.analytics.tests.factories import OHLCVFactory
from apps.market_data.models import Asset

# --[ تحسين: تعريف تكوين اختبار مركزي لاستخدامه في جميع الاختبارات ]--
TEST_ANALYTICS_CONFIG = {
    "HURST_WINDOW": 100,
    "ATR_WINDOW": 14,
    "REGIME_THRESHOLDS": {
        "TRENDING": 0.55,
        "MEAN_REVERTING": 0.45,
        "VOLATILITY_ATR_PCTL": 0.75,
    },
}

def generate_fbm_series(n: int, H: float) -> pd.Series:
    """Helper to generate Fractional Brownian Motion series."""
    ts = fbm.fbm(n=n - 1, hurst=H)
    return pd.Series(ts)


@pytest.mark.django_db
class TestOHLCVLoader:
    def test_load_dataframe_success(self):
        # ... (هذا الاختبار لم يتغير ويعمل بشكل صحيح)
        asset = Asset.objects.create(symbol="TEST")
        for i in range(200):
            OHLCVFactory(
                asset=asset, timestamp=timezone.now() - datetime.timedelta(hours=i)
            )
        loader = OHLCVLoader()
        df = loader.load_dataframe(
            asset, "H1", timezone.now() - datetime.timedelta(days=10), timezone.now()
        )
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df) >= settings.ANALYTICS_CONFIG.get("HURST_WINDOW", 100)

    def test_load_dataframe_no_data(self):
        # ... (هذا الاختبار لم يتغير ويعمل بشكل صحيح)
        asset = Asset.objects.create(symbol="EMPTY")
        loader = OHLCVLoader()
        with pytest.raises(ValueError, match="No OHLCV data found"):
            loader.load_dataframe(
                asset, "H1", timezone.now() - datetime.timedelta(days=1), timezone.now()
            )


@pytest.mark.django_db
class TestFeatureEngineer:
    def test_add_indicators(self):
        df = pd.DataFrame(
            {
                "high": [1.1, 1.2, 1.3] * 10,
                "low": [0.9, 1.0, 1.1] * 10,
                "close": [1.0, 1.1, 1.2] * 10,
            }
        )
        # --[ تصحيح: حقن التبعية (config) مباشرة لعزل الاختبار ]--
        engineer = FeatureEngineer(config=TEST_ANALYTICS_CONFIG)
        df_featured = engineer.add_indicators(df)
        assert "atr" in df_featured.columns
        assert not df_featured["atr"].isnull().all()


@pytest.mark.django_db
class TestRegimeClassifier: 
    def test_classify_trending(self):
            """
            Tests that the classifier correctly identifies a TRENDING regime.
            """
            # --[ تصحيح: استخدام قاموس التكوين المحلي بدلاً من الإعدادات العامة ]--
            data = generate_fbm_series(
                n=TEST_ANALYTICS_CONFIG["HURST_WINDOW"], H=0.75
            )
            
            atr_data = np.full(len(data), 0.01)

            df = pd.DataFrame(
                {
                    "close": data,
                    "high": data + atr_data,
                    "low": data - atr_data,
                    "atr": atr_data,
                }
            )
            # --[ تصحيح: حقن التبعية (config) مباشرة لعزل الاختبار ]--
            classifier = RegimeClassifier(config=TEST_ANALYTICS_CONFIG)
            regime, confidence, meta = classifier.classify(df)

            assert regime == "TRENDING"
            assert meta["hurst"] > 0.55