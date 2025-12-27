# apps/analytics/tasks.py
"""
Celery tasks for analytics pipeline (final, production-ready).

Pipeline steps (per-symbol chain):
1. ingest_historical_data_task  - ensure OHLCV in DB
2. update_asset_regime_task     - compute and persist market regime
3. scan_for_candidate_patterns  - run screeners and persist PatternCandidate
4. verify_pattern_candidates_task - verify candidates (DTW) and persist VerifiedPattern
5. generate_feature_vectors_task - build & persist FeatureVector snapshots (new)
6. trigger_decision_manager     - hand off to decision manager (execution layer)

This module contains robust error handling, defensive checks, sensible logging,
and a newly added feature-vector generation task that persists per-candle
feature snapshots used by ML / backtesting.
"""
from __future__ import annotations

import datetime
import time
from typing import Any, Dict, List, Optional

import numpy as np
import structlog
from celery import chain, shared_task
from django.db import transaction
from django.utils import timezone

from apps.analytics.patterns import screeners
from apps.analytics.patterns.templates import PatternTemplates
from apps.analytics.patterns.verifiers.dtw_verifier import DTWVerifier
from apps.analytics.services import OHLCVLoader, run_regime_analysis_for_asset
from apps.analytics.features.pipeline import FeaturePipeline
from apps.market_data.models import Asset
from apps.market_data.tasks import ingest_historical_data_task, trigger_decision_manager
from apps.trading_core.models import PatternCandidate, VerifiedPattern, FeatureVector

logger = structlog.get_logger(__name__)


def _periods_for_template_length(template_len: int, timeframe: str) -> datetime.timedelta:
    """
    Return a timedelta corresponding to template_len - 1 periods of the provided timeframe.
    Supported timeframe formats: "M5", "H1", "D1", etc.
    """
    if not timeframe or len(timeframe) < 2:
        logger.warning("Invalid timeframe format, defaulting to hours", timeframe=timeframe)
        return datetime.timedelta(hours=template_len - 1)

    unit = timeframe[0].upper()
    try:
        value = int(timeframe[1:])
    except (ValueError, TypeError):
        logger.warning("Cannot parse timeframe numeric part, defaulting to 1", timeframe=timeframe)
        value = 1

    if unit == "M":
        return datetime.timedelta(minutes=value * (template_len - 1))
    if unit == "H":
        return datetime.timedelta(hours=value * (template_len - 1))
    if unit == "D":
        return datetime.timedelta(days=value * (template_len - 1))

    logger.warning("Unknown timeframe unit, defaulting to hours", timeframe=timeframe)
    return datetime.timedelta(hours=value * (template_len - 1))


@shared_task(name="apps.analytics.tasks.update_asset_regime_task", bind=True, max_retries=2)
def update_asset_regime_task(self, prev_result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Task: run regime analysis for a single asset/timeframe pair.
    Expects prev_result to contain 'symbol' and 'timeframe'.
    Returns prev_result unchanged for chaining.
    """
    symbol: Optional[str] = prev_result.get("symbol")
    timeframe: Optional[str] = prev_result.get("timeframe")
    logger.info("Step 2/6: Starting market regime update", symbol=symbol, timeframe=timeframe)

    if not symbol or not timeframe:
        logger.warning("Missing symbol/timeframe for regime update; skipping", received=prev_result)
        return prev_result

    try:
        asset = Asset.objects.get(symbol__iexact=symbol)
    except Asset.DoesNotExist:
        logger.error("Asset not found for regime analysis", symbol=symbol)
        return prev_result

    try:
        run_regime_analysis_for_asset(asset, timeframe)
        logger.info("Regime analysis finished", symbol=symbol, timeframe=timeframe)
    except Exception as exc:
        logger.exception("Regime analysis failed - scheduling retry", symbol=symbol, timeframe=timeframe, exc_info=True)
        # propagate to Celery retry handling
        raise self.retry(exc=exc)

    return prev_result


@shared_task(name="apps.analytics.tasks.scan_for_candidate_patterns", bind=True, max_retries=2)
def scan_for_candidate_patterns(self, prev_result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Task: scan loaded OHLCV dataframe for pattern candidates using screeners.
    Persist PatternCandidate objects (bulk_create with ignore_conflicts).
    """
    symbol: Optional[str] = prev_result.get("symbol")
    timeframe: Optional[str] = prev_result.get("timeframe")
    logger.info("Step 3/6: Starting pattern scan", symbol=symbol, timeframe=timeframe)

    if not symbol or not timeframe:
        logger.warning("Missing symbol/timeframe for scan; skipping", received=prev_result)
        return prev_result

    try:
        asset = Asset.objects.get(symbol__iexact=symbol)
    except Asset.DoesNotExist:
        logger.warning("Asset not found for pattern scan", symbol=symbol)
        return prev_result

    loader = OHLCVLoader()
    end_utc = timezone.now()
    start_utc = end_utc - datetime.timedelta(days=45)

    try:
        df = loader.load_dataframe(asset, timeframe, start_utc, end_utc)
    except ValueError as e:
        logger.warning("OHLCV loader returned an error", symbol=symbol, error=str(e))
        return prev_result
    except Exception:
        logger.exception("Unexpected error while loading OHLCV for scanning", symbol=symbol)
        return prev_result

    if df is None or df.empty:
        logger.info("No OHLCV data available for scanning", symbol=symbol, timeframe=timeframe)
        return prev_result

    all_candidates: List[Any] = []
    try:
        # Run each screener in isolation to avoid single failure killing the scan
        try:
            all_candidates.extend(screeners.find_engulfing(df))
        except Exception:
            logger.exception("Screener 'find_engulfing' failed", symbol=symbol)

        try:
            all_candidates.extend(screeners.find_doji(df))
        except Exception:
            logger.exception("Screener 'find_doji' failed", symbol=symbol)

        try:
            all_candidates.extend(screeners.find_ma_crossover(df))
        except Exception:
            logger.exception("Screener 'find_ma_crossover' failed", symbol=symbol)

    except Exception:
        logger.exception("Unexpected failure while running screeners", symbol=symbol)

    if not all_candidates:
        logger.info("No new pattern candidates found", symbol=symbol)
        return prev_result

    candidate_objects: List[PatternCandidate] = []
    for c in all_candidates:
        try:
            candidate_objects.append(
                PatternCandidate(
                    asset=asset,
                    timestamp=c.timestamp,
                    pattern_type=c.pattern_type,
                    confidence=getattr(c, "confidence", None),
                    meta=getattr(c, "meta", None),
                )
            )
        except Exception:
            logger.exception("Failed to build PatternCandidate object from screener result", symbol=symbol, candidate_repr=repr(c))

    if not candidate_objects:
        logger.info("No valid PatternCandidate objects prepared for bulk create", symbol=symbol)
        return prev_result

    try:
        with transaction.atomic():
            created = PatternCandidate.objects.bulk_create(candidate_objects, ignore_conflicts=True)
        persisted = len(created) if created is not None else 0
        logger.info("Pattern scan task completed", symbol=symbol, found=len(all_candidates), persisted=persisted)
    except Exception:
        logger.exception("Bulk create for PatternCandidate failed", symbol=symbol)

    return prev_result


@shared_task(name="apps.analytics.tasks.verify_pattern_candidates_task", bind=True, max_retries=2)
def verify_pattern_candidates_task(self, prev_result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Task: verify unverified PatternCandidate instances using DTWVerifier and persist VerifiedPattern.
    Instrumented to record execution duration.
    """
    start_time = time.perf_counter()
    symbol: Optional[str] = prev_result.get("symbol")
    timeframe: Optional[str] = prev_result.get("timeframe")
    logger.info("Step 4/6: Starting pattern verification", symbol=symbol, timeframe=timeframe)

    if not symbol or not timeframe:
        logger.warning("Missing symbol/timeframe for verification task.", received=prev_result)
        return prev_result

    try:
        asset = Asset.objects.get(symbol__iexact=symbol)
    except Asset.DoesNotExist:
        logger.warning("Asset not found for verification", symbol=symbol)
        return prev_result

    # Load window large enough to cover templates
    end_utc = timezone.now()
    start_utc = end_utc - datetime.timedelta(days=10)

    try:
        df = OHLCVLoader().load_dataframe(asset, timeframe, start_utc, end_utc)
    except ValueError as e:
        logger.warning("OHLCV load failed for verification", symbol=symbol, error=str(e))
        return prev_result
    except Exception:
        logger.exception("Unexpected error loading OHLCV for verification", symbol=symbol)
        return prev_result

    if df is None or df.empty:
        logger.info("No OHLCV data available for verification", symbol=symbol, timeframe=timeframe)
        return prev_result

    min_index = df.index.min()
    candidates = PatternCandidate.objects.filter(
        asset=asset,
        timestamp__gte=min_index,
        verification__isnull=True
    ).order_by("timestamp")

    if not candidates.exists():
        duration = time.perf_counter() - start_time
        logger.info("No unverified candidates found to process", symbol=symbol, duration_seconds=round(duration, 2))
        return prev_result

    verified_instances: List[VerifiedPattern] = []
    for c in candidates:
        try:
            template = PatternTemplates.get_template(c.pattern_type)
            if template is None:
                logger.debug("No template found for candidate pattern type; skipping", candidate_id=c.id, pattern_type=c.pattern_type)
                continue

            span = _periods_for_template_length(len(template), timeframe)
            segment_start = c.timestamp - span
            segment_end = c.timestamp

            mask = (df.index >= segment_start) & (df.index <= segment_end)
            price_segment_series = df.loc[mask, "close"]

            if price_segment_series.empty:
                logger.debug("Price segment empty for candidate; skipping", candidate_id=c.id, start=segment_start, end=segment_end)
                continue

            price_segment = price_segment_series.to_numpy(dtype=float)

            confidence, distance = DTWVerifier.verify(price_segment, template)

            logger.debug("DTW verification result", candidate_id=c.id, confidence=confidence, distance=distance)

            # threshold used by original code (0.1)
            if confidence is not None and confidence > 0.1:
                verified_instances.append(
                    VerifiedPattern(
                        candidate=c,
                        verifier_type="DTW",
                        confidence=float(confidence),
                        meta={"dtw_distance": float(distance) if distance is not None else None},
                    )
                )
        except Exception:
            logger.exception("Failed to verify candidate", candidate_id=getattr(c, "id", None))

    if not verified_instances:
        duration = time.perf_counter() - start_time
        logger.info("Pattern verification task completed - nothing verified", symbol=symbol, verified_count=0, duration_seconds=round(duration, 2))
        return prev_result

    try:
        with transaction.atomic():
            created = VerifiedPattern.objects.bulk_create(verified_instances, ignore_conflicts=True)
            persisted = len(created) if created is not None else len(verified_instances)
    except Exception:
        logger.exception("Bulk create for VerifiedPattern failed", symbol=symbol)
        persisted = 0

    duration = time.perf_counter() - start_time
    logger.info("Pattern verification task completed", symbol=symbol, verified_count=persisted, duration_seconds=round(duration, 2))
    return prev_result


@shared_task(name="apps.analytics.tasks.generate_feature_vectors_task", bind=True, max_retries=2)
def generate_feature_vectors_task(self, prev_result: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Task: build and persist FeatureVector snapshots from OHLCV + feature pipeline.
    This stores the feature dict per timestamp for later ML/backtesting usage.
    """
    symbol: Optional[str] = prev_result.get("symbol")
    timeframe: Optional[str] = prev_result.get("timeframe")
    logger.info("Step 5/6: Starting feature vector generation", symbol=symbol, timeframe=timeframe)

    if not symbol or not timeframe:
        logger.warning("Missing symbol/timeframe for feature generation; skipping", received=prev_result)
        return prev_result

    try:
        asset = Asset.objects.get(symbol__iexact=symbol)
    except Asset.DoesNotExist:
        logger.warning("Asset not found for feature generation", symbol=symbol)
        return prev_result

    # Load OHLCV window to compute features
    end_utc = timezone.now()
    start_utc = end_utc - datetime.timedelta(days=45)
    try:
        ohlcv_df = OHLCVLoader().load_dataframe(asset, timeframe, start_utc, end_utc)
    except ValueError as e:
        logger.warning("Cannot generate features due to data issue", symbol=symbol, error=str(e))
        return prev_result
    except Exception:
        logger.exception("Unexpected error loading OHLCV for features", symbol=symbol)
        return prev_result

    if ohlcv_df is None or ohlcv_df.empty:
        logger.info("No OHLCV data available for feature generation", symbol=symbol)
        return prev_result

    # Build features DataFrame via FeaturePipeline (expected to return DataFrame indexed by timestamp)
    try:
        features_df = FeaturePipeline.build_feature_dataframe(symbol, ohlcv_df)
    except Exception:
        logger.exception("FeaturePipeline failed to build feature dataframe", symbol=symbol)
        return prev_result

    if features_df is None or features_df.empty:
        logger.info("FeaturePipeline returned empty dataframe; nothing to persist", symbol=symbol)
        return prev_result

    feature_vectors = []
    for timestamp, row in features_df.iterrows():
        try:
            # Convert numpy scalars and booleans to native python types for JSON storage
            feature_dict = {}
            for k, v in row.to_dict().items():
                if isinstance(v, (np.floating, np.integer)):
                    feature_dict[k] = float(v)
                elif isinstance(v, (np.bool_)):
                    feature_dict[k] = bool(v)
                else:
                    # allow None and other JSON serializable items
                    feature_dict[k] = v
            feature_vectors.append(
                FeatureVector(asset=asset, timestamp=timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp, features=feature_dict)
            )
        except Exception:
            logger.exception("Failed to convert feature row to FeatureVector", symbol=symbol, index=timestamp)

    if not feature_vectors:
        logger.info("No feature vectors were generated.", symbol=symbol)
        return prev_result

    # Persist feature vectors using update_or_create inside a transaction (DB-agnostic)
    try:
        with transaction.atomic():
            for fv in feature_vectors:
                FeatureVector.objects.update_or_create(
                    asset=fv.asset, timestamp=fv.timestamp, defaults={"features": fv.features}
                )
    except Exception:
        logger.exception("Failed to persist FeatureVectors", symbol=symbol)
        return prev_result

    logger.info("Feature vector generation task completed", symbol=symbol, count=len(feature_vectors))
    return prev_result


@shared_task(name="apps.analytics.tasks.main_analysis_loop_task")
def main_analysis_loop_task(symbols: List[str], timeframe: str) -> None:
    """
    Orchestrator task that queues a chain of tasks per symbol.
    Chain now includes feature vector generation as step 5/6.
    """
    logger.info("--- Starting Main Analysis Loop ---", symbols=symbols, timeframe=timeframe)
    now = timezone.now()
    for symbol in symbols:
        try:
            start_utc_ingest = now - datetime.timedelta(days=45)

            ingest_sig = ingest_historical_data_task.s(
                symbol=symbol,
                timeframe_str=timeframe,
                start_utc=start_utc_ingest,
                end_utc=now,
            )
            regime_sig = update_asset_regime_task.s()
            scan_sig = scan_for_candidate_patterns.s()
            verify_sig = verify_pattern_candidates_task.s()
            features_sig = generate_feature_vectors_task.s()
            decision_sig = trigger_decision_manager.s()

            workflow = chain(ingest_sig, regime_sig, scan_sig, verify_sig, features_sig, decision_sig)
            workflow.apply_async()
            logger.info("Workflow chain queued", symbol=symbol, timeframe=timeframe)
        except Exception:
            logger.exception("Failed to queue workflow chain for symbol", symbol=symbol)

    logger.info("--- All workflow chains for the loop have been queued ---")
