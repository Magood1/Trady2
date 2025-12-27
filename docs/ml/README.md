# Trady2 ML Pipeline (Sprint 4)

This document outlines the machine learning pipeline for training, evaluating, and serving prediction models.

## 1. How it Works

The pipeline uses `FeatureVector` data to predict short-term upward price movements.

- **Target Definition**: The target is binary. It's `1` if the price `N` hours in the future is `X%` higher than the current price. `N` (`horizon_steps`) and `X` (`return_threshold`) are configured in `configs/ml.example.yaml`.
- **Training**: We use a `LightGBM` classifier, trained on historical feature vectors. The process uses a time-based train/test split to prevent data leakage from the future.
- **Versioning**: Each trained model is saved (`.joblib`), hashed (SHA256), and registered in the `ModelRegistry`. Models must be manually activated via the Django Admin to be used by the prediction API.
- **Backtesting**: A vectorized backtester simulates a simple "long-only" strategy. It enters a trade on the `open` of the next bar if the model's prediction exceeds a threshold and holds for a fixed duration.
- **Prediction**: A live API endpoint (`/api/v1/analytics/predict/`) uses the currently active model to generate real-time predictions for an asset.

## 2. How to Run

### Training a New Model
1.  Customize `configs/ml.example.yaml` with your desired parameters.
2.  Ensure you have sufficient `FeatureVector` data for the target asset.
3.  Run the training command:
    ```bash
    python manage.py train_model --config configs/ml.example.yaml
    ```
4.  Go to the Django Admin (`/admin/mlops/modelregistry/`), select the new model, and use the "Activate selected model" action from the dropdown.

### Running a Backtest
1.  Ensure you have a trained model. Note its version string from the training output or the admin panel (e.g., `1.0.0+20251101_120000`).
2.  Run the backtesting command:
    ```bash
    python manage.py backtest_model --version "1.0.0+20251101_120000" --config configs/ml.example.yaml --output report
    ```
3.  This will generate `report.html` (interactive plot) and `report.json` (metrics summary).

## 3. Risks and Considerations
- **Overfitting**: The model may perform well on historical data but fail in live markets. Always treat backtest results with caution.
- **Model Drift**: Market dynamics change. Models must be retrained periodically to remain effective.
- **Backtester Simplifications**: The current backtester holds for a fixed duration and does not implement dynamic stop-loss or take-profit, which can significantly affect performance. This should be a priority for future sprints.
- **Data Leakage**: We have taken steps to prevent lookahead bias (using next bar's open for entry, time-based splits). However, complex features could still inadvertently leak information. Constant vigilance is required.





## 4. Environment and CI/CD

### Docker
The included `Dockerfile` is configured to install all necessary system dependencies for building `lightgbm`, such as `cmake` and `build-essential`. The application can be built and run using `docker-compose`.

### CI/CD
For CI pipelines (like GitHub Actions), ensure the runner environment has these build tools or use the provided Docker image as the base for your testing stage. A typical step in your CI workflow might look like this:

```yaml
- name: Install System Dependencies
  run: sudo apt-get update && sudo apt-get install -y build-essential cmake