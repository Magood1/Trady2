# Analytics Layer (Sprint 1)

This document describes the first intelligent layer of the Trady2 system: the Analytics Layer, focusing on the Market Regime Analyzer.

## Overview

The goal of this layer is to automatically classify the current state (or "regime") of the market for any given asset. This provides crucial context for downstream decision-making models. For example, a trend-following strategy should only be active in a `TRENDING` regime.

## Core Components

1.  **Hurst Exponent (`hurst_analyzer.py`):**
    -   **Purpose:** Measures the long-term memory of a time series to determine if it's trending, mean-reverting, or random.
    -   **Implementation:** Uses the Rescaled Range (R/S) method.
    -   **Parameters:** The primary parameter is `HURST_WINDOW` (default: 100) in `settings.py`. A larger window provides a more stable, long-term estimate but is slower to react to changes.

2.  **Average True Range (ATR) (`atr_analyzer.py`):**
    -   **Purpose:** Measures market volatility.
    -   **Parameters:** `ATR_WINDOW` (default: 14).

3.  **Regime Classifier (`services.py`):**
    -   **Logic:** A simple, rule-based classifier that uses Hurst and ATR to determine the regime.
    -   **Rules (configurable in `settings.py`):**
        -   If ATR is above its historical percentile (`VOLATILITY_ATR_PCTL`), the regime is `HIGH_VOLATILITY`.
        -   Else, if Hurst > `TRENDING` threshold, the regime is `TRENDING`.
        -   Else, if Hurst < `MEAN_REVERTING` threshold, the regime is `MEAN_REVERTING`.
        -   Otherwise, it's `RANDOM`.

## API Usage

You can retrieve the latest calculated regime for any symbol via a GET request.

**Endpoint:** `/api/v1/analytics/regime/{symbol}/`

**Example Request:**
```bash
# Get the most recent regime for EURUSD
curl -X GET http://localhost:8000/api/v1/analytics/regime/EURUSD/

# Get the regime for EURUSD as of a specific time
curl -X GET "http://localhost:8000/api/v1/analytics/regime/EURUSD/?at=2025-01-15T12:00:00Z"

