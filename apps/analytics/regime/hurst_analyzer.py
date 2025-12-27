# apps/analytics/regime/hurst_analyzer.py
import numpy as np
import pandas as pd
from typing import Optional

# optional import of hurst lib
try:
    from hurst import compute_Hc  # type: ignore
    _HURST_LIB_AVAILABLE = True
except Exception:
    _HURST_LIB_AVAILABLE = False


def _hurst_dfa(ts: np.ndarray, min_window: int = 4, max_window: Optional[int] = None,
               n_windows: int = 20) -> Optional[float]:
    """
    DFA estimator. Expects a (approximately) stationary input (e.g. returns or fGn).
    Returns None on failure.
    """
    N = len(ts)
    if N < 50:
        return None

    if max_window is None:
        max_window = max(min_window + 1, N // 4)

    # profile (integrated series) — DFA works on integrated (profile) of stationary signal
    profile = np.cumsum(ts - np.mean(ts))

    # log-spaced window sizes
    sizes = np.unique(
        np.floor(
            np.logspace(np.log10(min_window), np.log10(max_window), num=n_windows)
        ).astype(int)
    )
    sizes = sizes[sizes >= 4]
    if len(sizes) < 3:
        return None

    Fs = []
    Ss = []
    for s in sizes:
        n_segments = N // s
        if n_segments < 3:
            continue

        rms_per_seg = []
        for i in range(n_segments):
            seg = profile[i * s: (i + 1) * s]
            if seg.size < 2:
                continue
            x = np.arange(seg.size)
            try:
                p = np.polyfit(x, seg, 1)
                trend = np.polyval(p, x)
                detrended = seg - trend
                rms = np.sqrt(np.mean(detrended ** 2))
                if np.isfinite(rms) and rms > 0.0:
                    rms_per_seg.append(rms)
            except Exception:
                continue

        if len(rms_per_seg) < max(3, n_segments // 2):
            continue

        F_s = np.sqrt(np.mean(np.array(rms_per_seg) ** 2))
        if not np.isfinite(F_s) or F_s <= 0.0:
            continue

        Fs.append(F_s)
        Ss.append(s)

    if len(Fs) < 3:
        return None

    logS = np.log10(Ss)
    logF = np.log10(Fs)
    try:
        slope, intercept = np.polyfit(logS, logF, 1)
        H = float(slope)  # DFA slope ~ H for stationary increments
        if not np.isfinite(H):
            return None
        # sanity clamp but be conservative; we will vet externally
        if H < -1.0 or H > 3.0:
            return None
        return H
    except Exception:
        return None


def _is_effectively_stationary(ts: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Lightweight heuristic to decide if `ts` is stationary enough for DFA directly.
    We check whether variance grows strongly with time (non-stationary price path).
    """
    if len(ts) < 100:
        return False
    # slope of linear fit vs std of residuals
    x = np.arange(len(ts))
    try:
        p = np.polyfit(x, ts, 1)
        trend = np.polyval(p, x)
        resid = ts - trend
        # if trend magnitude relative to residual std is small -> approximately stationary
        if np.std(resid) <= 0:
            return False
        rel = abs(p[0]) / (np.std(resid) + tol)
        # heuristic threshold: if trend is more than 1e-3 of residual std per sample -> non-stationary
        return rel < 1e-3
    except Exception:
        return False


def hurst_exponent(series: pd.Series, method: str = "auto") -> float:
    """
    Robust Hurst estimator.

    method: "auto" | "hurst_lib" | "dfa"
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas.Series")

    n = len(series)
    if n < 100:
        return np.nan

    ts = series.to_numpy(dtype=float)
    if not np.all(np.isfinite(ts)):
        ts = ts[np.isfinite(ts)]
        if len(ts) < 100:
            return np.nan

    if np.std(ts) < 1e-9:
        return 0.5

    # 1) try hurst library if available and allowed
    if method in ("auto", "hurst_lib") and _HURST_LIB_AVAILABLE:
        try:
            H, c, data = compute_Hc(ts, kind="price", simplified=True)
            if np.isfinite(H) and 0.0 <= H <= 1.0:
                return float(H)
            print(f"[hurst_analyzer] hurst.compute_Hc returned invalid H: {H}")
        except Exception as e:
            print(f"[hurst_analyzer] hurst.compute_Hc failed with exception: {e}")

        if method == "hurst_lib":
            return np.nan

    # 2) DFA fallback(s): try sensible inputs and pick the most plausible result.
    candidates = []

    # if ts seems stationary enough, try DFA directly on ts (treat ts as increments)
    if _is_effectively_stationary(ts):
        H1 = _hurst_dfa(ts)
        if H1 is not None:
            candidates.append(("dfa_direct", H1))

    # try DFA on first differences (returns / increments) — safe for price inputs
    diffs = np.diff(ts)
    if len(diffs) >= 50:
        H2 = _hurst_dfa(diffs)
        if H2 is not None:
            candidates.append(("dfa_diff", H2))

    # choose best candidate: prefer values in (0.05,0.95) and not extreme
    def score_candidate(h_val: float) -> float:
        if h_val is None or not np.isfinite(h_val):
            return -np.inf
        # prefer near 0.5 (neutral), but we only need plausibility: penalize extremes
        return -abs(h_val - 0.5)  # higher is better

    if candidates:
        # sort by score descending
        cand_sorted = sorted(candidates, key=lambda t: score_candidate(t[1]), reverse=True)
        best_name, best_h = cand_sorted[0]
        # final clamp to [0,1]
        best_h = float(max(0.0, min(1.0, best_h)))
        print(f"[hurst_analyzer] using {best_name} H={best_h}")
        return best_h

    # 3) Last resort: give neutral 0.5 rather than break the pipeline
    return 0.5
