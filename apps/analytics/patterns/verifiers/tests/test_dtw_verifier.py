# apps/analytics/patterns/verifiers/tests/test_dtw_verifier.py
import numpy as np
import pytest

from apps.analytics.patterns.templates import PatternTemplates
from apps.analytics.patterns.verifiers.dtw_verifier import DTWVerifier, _normalize_series

def test_normalize_series():
    series = np.array([10, 20, 30, 40, 50], dtype=float)
    normalized = _normalize_series(series)
    assert np.allclose(normalized, [0.0, 0.25, 0.5, 0.75, 1.0])
    flat_series = np.array([10, 10, 10, 10], dtype=float)
    normalized_flat = _normalize_series(flat_series)
    assert np.allclose(normalized_flat, [0.0, 0.0, 0.0, 0.0])

def test_dtw_verifier_good_match():
    """
    Tests that a segment closely matching a template gets a high confidence score.
    This test is now deterministic.
    """
    np.random.seed(42)
    
    price_segment = np.concatenate([
        np.linspace(100, 50, 10),
        np.linspace(55, 105, 10)
    ]) + np.random.normal(0, 2, 20)

    template = PatternTemplates.V_SHAPE_RECOVERY
    confidence, distance = DTWVerifier.verify(price_segment, template)
    
    assert confidence > 0.8
    assert distance < 5.0

def test_dtw_verifier_bad_match():
    """
    Tests that a segment with an opposing shape (A-shape vs V-shape)
    gets a very low confidence score.
    """
    # --[ تصحيح: استخدام شكل "A-shape" كحالة اختبار سيئة واضحة بدلاً من الضوضاء العشوائية ]--
    # هذا الشكل هو عكس قالب V-shape تمامًا، مما يضمن مسافة DTW كبيرة.
    price_segment_a_shape = np.concatenate([
        np.linspace(50, 100, 10),
        np.linspace(100, 50, 10)
    ])
    
    template = PatternTemplates.V_SHAPE_RECOVERY
    confidence, distance = DTWVerifier.verify(price_segment_a_shape, template)
    
    assert confidence < 0.5
    assert distance > 5.0 # نتوقع مسافة كبيرة جدًا