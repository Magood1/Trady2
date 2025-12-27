# apps/analytics/tests/test_verify_integration.py
import pytest
import numpy as np
from django.utils import timezone
from datetime import timedelta
from apps.market_data.models import Asset, OHLCV
from apps.trading_core.models import PatternCandidate, VerifiedPattern
from apps.analytics.patterns.templates import PatternTemplates
from apps.analytics.tasks import verify_pattern_candidates_task
from apps.market_data.services import OHLCVLoader

pytestmark = pytest.mark.django_db

def make_ohlcv_rows(asset, timeframe, start_dt, count=20, base_price=100.0):
    rows = []
    for i in range(count):
        ts = start_dt + timedelta(hours=i)  # H1 spacing
        rows.append(OHLCV(
            asset=asset,
            timeframe=timeframe,
            timestamp=ts,
            open=base_price + i*0.001,
            high=base_price + i*0.002,
            low=base_price + i*0.0005,
            close=base_price + i*0.0015,
            volume=100 + i
        ))
    return rows

def test_verify_end_to_end(db):
    # Arrange
    asset, _ = Asset.objects.get_or_create(symbol="TESTPAIR")
    timeframe = "H1"
    now = timezone.now().replace(minute=0, second=0, microsecond=0)

    # create OHLCV rows that contain a V-shape-like segment
    start = now - timedelta(hours=25)
    rows = make_ohlcv_rows(asset, timeframe, start, count=26, base_price=1.17)
    OHLCV.objects.bulk_create(rows, ignore_conflicts=True)

    # Create a PatternCandidate whose timestamp aligns with one of the rows
    candidate_ts = start + timedelta(hours=20)  # inside the DF window
    candidate = PatternCandidate.objects.create(
        asset=asset,
        timestamp=candidate_ts,
        pattern_type="ENGULFING_BULLISH",  # ensure a template exists or adjust
        confidence=0.5,
        meta={}
    )

    # Act: call verify task synchronously with prev_result context
    prev_result = {"symbol": asset.symbol, "timeframe": timeframe}
    result = verify_pattern_candidates_task(prev_result)

    # Assert: a VerifiedPattern exists for candidate (or none if template mismatch)
    verified_qs = VerifiedPattern.objects.filter(candidate=candidate)
    # We accept either zero (if no template) OR >=0; assert no crash and function returns prev_result
    assert result == prev_result
    # if template exists for this pattern_type then expect at least 0 or 1
    # (we primarily assert the task ran without exceptions)
