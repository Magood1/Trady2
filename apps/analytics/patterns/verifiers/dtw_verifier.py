# apps/analytics/patterns/verifiers/dtw_verifier.py
from typing import Tuple
import numpy as np
import structlog
from fastdtw import fastdtw
from django.conf import settings

logger = structlog.get_logger(__name__)

def _normalize_series(series: np.ndarray) -> np.ndarray:
    if series.size == 0: return series
    min_val, max_val = float(np.min(series)), float(np.max(series))
    return np.zeros_like(series, dtype=float) if max_val == min_val else (series - min_val) / (max_val - min_val)

class DTWVerifier:
    @staticmethod
    def verify(price_segment: np.ndarray, template: np.ndarray) -> Tuple[float, float]:
        seg, tpl = np.asarray(price_segment, dtype=float).flatten(), np.asarray(template, dtype=float).flatten()
        if seg.size < 2 or tpl.size < 2: return 0.0, float('inf')

        seg_normalized, tpl_normalized = _normalize_series(seg), _normalize_series(tpl)
        try:
            distance, path = fastdtw(seg_normalized, tpl_normalized, dist=lambda a, b: abs(a - b))
        except Exception:
            logger.exception("An unexpected error occurred in fastdtw.")
            return 0.0, float('inf')

        avg_distance = float(distance) / max(1.0, len(path))
        k = settings.ANALYTICS_CONFIG.get("DTW_SENSITIVITY_K", 4.0)
        confidence = max(0.0, min(1.0, 1.0 / (1.0 + k * avg_distance)))
        return round(confidence, 4), round(float(distance), 4)