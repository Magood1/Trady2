# Trading Logic and Risk Management (Sprint 3)

This document outlines the integrated trading logic, which combines analysis from previous sprints with a critical new layer: Risk Management.

## Decision Pipeline

The system now follows a sequential pipeline to generate a trading signal:

1.  **Regime Check:** The `DecisionManager` first checks the current `MarketRegime`.
    -   **Rule:** It only proceeds if the regime is `TRENDING`. All other regimes are ignored for now.

2.  **Pattern Check:** It then looks for the most recent `PatternCandidate` in the database.
    -   **Rule:** It specifically looks for bullish patterns (e.g., `ENGULFING_BULLISH`) that occurred recently.

3.  **Confirmation Filter:** Before accepting the pattern, it applies a confirmation filter.
    -   **Rule:** The Relative Strength Index (RSI) must **not** be in the overbought zone (> 70) for a buy signal.

4.  **Risk Management:** If all checks pass, the `RiskManager` is invoked.
    -   **Stop Loss:** Calculated as `2 * ATR` below the entry price for a buy signal.
    -   **Take Profit:** Calculated to achieve a `1:1.5` Risk-to-Reward ratio.
    -   **Position Size:** Calculated based on a fixed fractional risk (`1%` of account balance) and the stop loss distance.

5.  **Signal Persistence:** Only if all the above steps succeed is a `TradingSignal` object created and saved to the database. This object represents a fully vetted, ready-to-execute trade.

## Triggering Mechanism

A new Celery task, `trigger_decision_manager`, has been created. This task simulates a real-time listener. In a live environment, after each new candle is ingested from MT5, a task like this would be dispatched to run the analysis pipeline.