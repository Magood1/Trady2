Trady2: Quantitative Research & Execution Ecosystem

Trady2 is an enterprise-grade, modular monolith framework designed for Quantitative Finance R&D and Algorithmic Trading.

Built with a focus on Software Engineering rigor, it bridges the gap between raw financial data and actionable ML insights. The system is architected to handle the full MLOps lifecycle: from high-frequency data ingestion and feature engineering to model training, strict backtesting protocols (preventing look-ahead bias), and live execution orchestration.

Note: This project serves as a comprehensive engineering showcase demonstrating Domain-Driven Design (DDD), Event-Driven Architecture, and Production-Grade MLOps pipelines.

Technical Highlights
1. Robust Backend Architecture

Modular Monolith with DDD: The codebase is strictly separated into bounded contexts:

market_data: Data ingestion, normalization, and persistence.

analytics: Feature engineering pipelines and ML model lifecycle.

trading_core: Strategy logic, risk management, and order execution.

Event-Driven Design: Utilizes Celery and Redis to handle asynchronous tasks (e.g., fetching OHLCV data, triggering model inference) without blocking the main application thread.

Database Optimization: Engineered for time-series efficiency using PostgreSQL patterns (compatible with TimescaleDB).

2. Advanced MLOps & Quantitative Research

Strict Validation Protocols: Implements Time-Series Split and Triple Barrier Method (TBM) labeling to eliminate look-ahead bias and ensure statistical significance.

Dynamic Feature Pipeline: A centralized FeaturePipeline class ensures consistency between training and inference environments, preventing training-serving skew.

Meta-Labeling Strategy: Uses a primary technical filter combined with a secondary LightGBM meta-model to filter out false positives and improve execution precision.

3. DevOps & Maintainability

Infrastructure as Code: Fully containerized using Docker and Docker Compose for reproducible environments.

Developer Experience (DX): Extensive Makefile automation for setup, testing, and linting.

Code Quality: Enforced type hinting (typing), static analysis (mypy), and formatting (black/flake8).

Architecture Overview

The system follows a linear data flow pipeline designed for stability and auditability:

Ingestion Layer: The MT5Connector fetches historical/live data via a fault-tolerant retry mechanism (tenacity).

Persistence Layer: Data is normalized into OHLCV models and stored.

Analytics Layer:

Feature Engineering: Raw data is transformed into vector inputs (Volatility, RSI, ADX, etc.) using a vectorized pipeline.

Model Registry: Trained models are versioned, hashed, and stored via ModelRegistry (MLOps).

Decision Layer: The DecisionManager aggregates market regime context, strategy signals, and ML probability scores to generate a TradingSignal.

Execution Layer: A CircuitBreaker pattern protects capital before the ExecutionManager dispatches orders to the broker.

Quick Start for Engineers

This project assumes a standard Python/Docker environment.

Prerequisites

Docker & Docker Compose

Python 3.10+ (for local development)

Setup & Run

We use a Makefile to standardize development tasks.

Clone and Configure:
git clone <repo_url>
cd trady2
cp .env.example .env

Build the Stack:
make up

Run Migrations:
make migrate

Ingest Sample Data (Verification):
# Fetches 1 year of H1 data for XAUUSD to verify pipeline health
python manage.py ingest_historical_data XAUUSD H1 --days=365


Code Showcase

A glimpse into the engineering standards applied in this project.

1. Vectorized Feature Engineering (Pandas)

Demonstrating efficient data manipulation and Type Hinting.
class FeaturePipeline:
    @staticmethod
    def build_feature_dataframe(symbol: str, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Single Source of Truth for Feature Engineering.
        Ensures strict alignment between training and inference.
        """
        df = ohlcv_df.copy()
        
        # Vectorized calculation of Volatility (Log Returns Std Dev)
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['vol_std'] = df['log_ret'].rolling(window=20).std()
        
        # Context-Aware Indicators
        adx_ind = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['adx'] = adx_ind.adx()

        # Data Cleaning & Type Safety
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        
        return df[['vol_std', 'adx', 'rsi']]

2. The Decision Manager (Domain Logic)

Demonstrating encapsulation and risk controls.
class DecisionManager:
    def run_analysis(self) -> Optional[TradingSignal]:
        # 1. Base Strategy Filter
        if not self._check_trend_pullback():
            return None

        # 2. ML Meta-Model Gatekeeper
        model, registry = get_active_model()
        ml_prob = model.predict_proba(self.latest_features)[0, 1]
        
        if ml_prob < self.ML_THRESHOLD:
            logger.info(f"Signal rejected by ML model (Prob: {ml_prob:.2f})")
            return None

        # 3. Dynamic Risk Sizing (Volatility Adjusted)
        stop_dist = self.current_volatility * Decimal("2.0")
        position_size = self.risk_manager.calculate_size(stop_dist)

        return TradingSignal.objects.create(...)

Development Methodology

This project adheres to strict software engineering practices suitable for distributed teams.

Testing Strategy: Unit tests and integration tests are written using pytest. We mock external dependencies (like the MT5 connection) to ensure tests are deterministic.

make test

Static Analysis: mypy is used for static type checking to catch errors at build time, and flake8 ensures PEP8 compliance.

CI/CD Ready: The project structure is designed to plug easily into GitHub Actions or Jenkins pipelines for automated testing and deployment.

🛠 Tech Stack

Language: Python 3.11

Framework: Django 5.0 (DRF)

Task Queue: Celery + Redis

Data Science: Pandas, NumPy, Scikit-learn, LightGBM, TA-Lib

Database: SQLite (Dev) / PostgreSQL (Prod ready)

Infrastructure: Docker, Docker Compose

🤝 Contribution

This is currently a private R&D repository. However, the architecture is designed to be extensible. Future modules planned include:

Sentiment Analysis Module: NLP on financial news feeds.

Reinforcement Learning Agent: For dynamic position sizing.

Engineered with precision.