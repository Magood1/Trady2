import fbm
import numpy as np
import pandas as pd
import pytest

from apps.analytics.regime.hurst_analyzer import hurst_exponent

# --- دالة مساعدة لإنشاء بيانات اختبار مثالية ---
def generate_fbm_series(n: int, H: float) -> pd.Series:
    """
    Generates a Fractional Brownian Motion series.
    We no longer need to add artificial noise because our algorithm is now robust.
    """
    ts = fbm.fbm(n=n - 1, hurst=H)
    return pd.Series(ts)
# ------------------------------------------------

def test_hurst_exponent_on_mean_reverting():
    """
    Tests Hurst on an anti-persistent series (H < 0.5).
    This series should revert to its mean.
    """
    series = generate_fbm_series(1000, 0.25)
    h = hurst_exponent(series)
    assert h < 0.5, f"Expected H < 0.5 for mean-reverting series, but got {h}"

def test_hurst_exponent_on_random_walk():
    """
    Tests Hurst on a geometric Brownian motion series (H = 0.5).
    This is a pure random walk.
    """
    series = generate_fbm_series(1000, 0.5)
    h = hurst_exponent(series)
    # We allow a tolerance margin around 0.5
    assert 0.4 < h < 0.6, f"Expected H to be around 0.5 for random walk, but got {h}"

def test_hurst_exponent_on_trending():
    """
    Tests Hurst on a persistent series (H > 0.5).
    This series has long-term memory and a tendency to trend.
    """
    series = generate_fbm_series(1000, 0.75)
    h = hurst_exponent(series)
    assert h > 0.5, f"Expected H > 0.5 for trending series, but got {h}"


import pandas as pd
from apps.analytics.regime.hurst_analyzer import hurst_exponent
import fbm

def test_hurst_fbm_trend_and_diff():
    s = pd.Series(fbm.fbm(n=1000, hurst=0.75))
    h = hurst_exponent(s)
    assert 0.6 < h < 0.9

def test_hurst_fbm_random_walk():
    s = pd.Series(fbm.fbm(n=1000, hurst=0.5))
    h = hurst_exponent(s)
    assert 0.4 < h < 0.6

def test_hurst_fbm_mean_reverting():
    s = pd.Series(fbm.fbm(n=1000, hurst=0.25))
    h = hurst_exponent(s)
    assert 0.1 < h < 0.45



# import numpy as np 
# import pandas as pd
# import pytest

# from apps.analytics.regime.hurst_analyzer import hurst_exponent


# def generate_random_walk(length: int) -> pd.Series:
#     return pd.Series(np.random.randn(length).cumsum())


# def generate_mean_reverting(length: int) -> pd.Series:
#     a = 0.9
#     x = np.zeros(length)
#     x[0] = 0
#     for t in range(1, length):
#         x[t] = a * x[t - 1] + np.random.randn()
#     return pd.Series(x)


# def generate_trending() -> pd.Series:
#     return pd.Series(np.arange(1000) + np.random.randn(1000) * 0.1)


# def test_hurst_exponent_on_random_walk():
#     series = generate_random_walk(1000)
#     h = hurst_exponent(series)
#     # For a random walk, H should be close to 0.5
#     assert 0.4 < h < 0.6


# def test_hurst_exponent_on_mean_reverting():
#     series = generate_mean_reverting(1000)
#     h = hurst_exponent(series)
#     # For a mean-reverting series, H should be < 0.5
#     assert h < 0.5


# def test_hurst_exponent_with_insufficient_data():
#     series = pd.Series(np.random.randn(50))
#     h = hurst_exponent(series)
#     assert np.isnan(h)
