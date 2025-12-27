# Pattern Scanning Layer (Sprint 2)

This document describes the second layer of the Trady2 system: the Pattern Scanning and Candidate Persistence layer.

## Overview

The goal of this layer is to perform a fast, deterministic scan of OHLCV data to identify potential trading patterns. These "candidates" are then stored in the database for further analysis or for use by decision-making modules.

## Core Components

1.  **Screeners (`screeners.py`):**
    -   **Purpose:** Contains vectorized, high-performance functions that search for specific patterns (e.g., Engulfing, Doji) in a pandas DataFrame.
    -   **Design:** Each screener is a pure function that takes a DataFrame and returns a list of `PatternCandidateData` objects. They do not interact with the database.

2.  **Pattern Candidate Model (`trading_core/models.py`):**
    -   **Purpose:** A Django model to persist detected pattern candidates.
    -   **Key Feature:** A `unique_together` constraint on `(asset, timestamp, pattern_type)` prevents duplicate entries.

3.  **Scanning Task (`analytics/tasks.py`):**
    -   **Purpose:** A Celery task (`scan_for_candidate_patterns`) that orchestrates the entire process: loads data, runs all screeners, and saves the results to the database.
    -   **Efficiency:** Uses `bulk_create` with `ignore_conflicts=True` for high-performance "upsert" semantics.

## Automation & Usage

The pattern scanning process is designed to be triggered automatically after new data is ingested.

**Manual Trigger (for testing):**
You can trigger a scan for a specific asset and timeframe using a Django management command or by calling the Celery task directly.

**Example Management Command (to be created in `market_data` app):**
```bash
python manage.py scan_patterns EURUSD H1 --days=30