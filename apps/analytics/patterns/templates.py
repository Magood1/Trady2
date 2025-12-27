# apps/analytics/patterns/templates.py
"""
Stores ideal pattern templates for verification algorithms (e.g., DTW).

Each template is a 1-D NumPy array representing the normalized price movement
shape of a pattern. Templates returned by `get_template` are guaranteed to be
numpy arrays with values scaled to the [0.0, 1.0] range (min-max normalization).
This helps make comparisons scale-invariant for distance-based verifiers.
"""

from __future__ import annotations

from typing import Optional, Dict
import numpy as np


class PatternTemplates:
    """A repository for ideal financial pattern shapes."""

    # -----------------------
    # Raw templates (readable / editable)
    # -----------------------
    # V-shape recovery: price falls then recovers (longer template)
    V_SHAPE_RECOVERY_RAW: np.ndarray = np.concatenate(
        [np.linspace(1, 0, 10), np.linspace(0.1, 1, 10)]
    )

    # Engulfing patterns (short templates)
    BULLISH_ENGULFING_RAW: np.ndarray = np.array([0.8, 0.7, 1.0])
    BEARISH_ENGULFING_RAW: np.ndarray = np.array([0.7, 0.8, 0.5])

    # Doji (represents indecision; close ~ open)
    DOJI_RAW: np.ndarray = np.array([0.5, 1.0, 0.0, 0.8, 0.55])

    # Moving-average cross templates (short)
    MA_CROSS_BULLISH_RAW: np.ndarray = np.array([0.5, 0.4, 0.3, 0.8, 1.0])
    MA_CROSS_BEARISH_RAW: np.ndarray = np.array([0.5, 0.6, 0.7, 0.2, 0.0])

    # -----------------------
    # Internal registry mapping canonical pattern_type -> raw array attribute
    # -----------------------
    _REGISTRY: Dict[str, np.ndarray] = {
        "V_SHAPE_RECOVERY": V_SHAPE_RECOVERY_RAW,
        "ENGULFING_BULLISH": BULLISH_ENGULFING_RAW,
        "ENGULFING_BEARISH": BEARISH_ENGULFING_RAW,
        "DOJI": DOJI_RAW,
        "MA_CROSS_BULLISH": MA_CROSS_BULLISH_RAW,
        "MA_CROSS_BEARISH": MA_CROSS_BEARISH_RAW,
    }

    # -----------------------
    # Utilities
    # -----------------------
    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """
        Min-max normalize a 1-D numpy array to range [0.0, 1.0].

        If the array is constant (max == min), returns an array of 0.5 values
        (neutral mid-point) to avoid degenerate zero-length sequences in some
        verifiers.
        """
        arr = np.asarray(arr, dtype=float).flatten()
        if arr.size == 0:
            return arr
        mn = arr.min()
        mx = arr.max()
        if np.isclose(mx, mn):
            # constant signal -> neutral mid-point
            return np.full_like(arr, fill_value=0.5, dtype=float)
        return (arr - mn) / (mx - mn)

    # -----------------------
    # Public API
    # -----------------------
    @classmethod
    def get_template(cls, pattern_type: str) -> Optional[np.ndarray]:
        """
        Return a normalized copy of the template array for the given pattern_type.

        Args:
            pattern_type: canonical pattern name, e.g. "ENGULFING_BULLISH", "DOJI"

        Returns:
            1-D numpy.ndarray with values scaled to [0.0, 1.0], or None if unknown.
        """
        if not pattern_type:
            return None

        key = pattern_type.strip().upper()
        raw = cls._REGISTRY.get(key)
        if raw is None:
            return None

        # Return a normalized copy to avoid accidental in-place mutation by callers
        return cls._normalize(raw).copy()

    @classmethod
    def list_templates(cls) -> Dict[str, int]:
        """
        Return a mapping of available template names to their lengths (number of points).

        Useful for introspection and for deciding window sizes when slicing price data.
        """
        return {name: int(np.asarray(arr).size) for name, arr in cls._REGISTRY.items()}

    @classmethod
    def add_custom_template(cls, name: str, template: np.ndarray) -> None:
        """
        Add a custom template to the registry at runtime.

        Note: the template is stored as provided (raw). `get_template` will normalize it.
        """
        if not name or template is None:
            raise ValueError("Both name and template must be provided")
        key = name.strip().upper()
        cls._REGISTRY[key] = np.asarray(template, dtype=float)

    @classmethod
    def remove_template(cls, name: str) -> bool:
        """
        Remove a template from the registry. Returns True if removed, False if not found.
        """
        key = name.strip().upper()
        return cls._REGISTRY.pop(key, None) is not None
