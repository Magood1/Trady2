from typing import Any

from django.conf import settings


class PatternConfigs:
    """
    Loads and provides access to pattern detection configurations.
    Defaults are sourced from Django settings, allowing for centralized management.
    """

    DEFAULT_CONFIG = {
        "ENGULFING_BODY_FACTOR": 0.8,
        "DOJI_THRESHOLD_RATIO": 0.1,
        "MA_CROSS_FAST_WINDOW": 5,
        "MA_CROSS_SLOW_WINDOW": 21,
        "FILTER_BY_REGIME": True,
        "ATR_FILTER_PERCENTILE": 0.9,  # Filter out top 10% volatile signals
    }

    @staticmethod
    def get_config() -> dict[str, Any]:
        """Returns the analytics configuration dictionary."""
        return getattr(settings, "ANALYTICS_CONFIG", {}).get(
            "PATTERNS", PatternConfigs.DEFAULT_CONFIG
        )