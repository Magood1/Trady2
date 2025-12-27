# apps/market_data/services.py
"""
OHLCV loading and ingestion services for market data.

Provides:
- OHLCVLoader: load OHLCV rows from DB into a tz-aware pandas.DataFrame (UTC index)
- ingest_ohlcv_data: fetch OHLCV from MT5Connector and persist into OHLCV model

Notes:
- All DataFrame indexes returned by OHLCVLoader.load_dataframe are guaranteed to be timezone-aware (UTC).
- ingest_ohlcv_data ensures timestamps saved to the DB are Python datetime objects.
- bulk_create is used with ignore_conflicts=True for performance when duplicates are expected.
"""

from __future__ import annotations

import datetime
from typing import Any, Optional

import pandas as pd
import structlog
from django.db import transaction
from django.utils import timezone

from apps.common.enums import Timeframe
from apps.market_data.connectors.mt5_connector import MT5Connector
from apps.market_data.models import Asset, OHLCV

logger = structlog.get_logger(__name__)


class OHLCVLoader:
    """
    Loader helper that reads OHLCV rows from the database and returns a pandas DataFrame.

    The returned DataFrame:
      - has index = timestamp (tz-aware, UTC)
      - columns: ['open','high','low','close','volume']
      - numeric price columns are floats

    Example:
        df = OHLCVLoader().load_dataframe(asset, "M5", start_utc, end_utc)
    """

    def load_dataframe(
        self,
        asset: Asset,
        timeframe: str,
        start_utc: datetime.datetime,
        end_utc: datetime.datetime,
    ) -> pd.DataFrame:
        """
        Load OHLCV rows for `asset`/`timeframe` between start_utc and end_utc (inclusive start, exclusive end).
        Ensures returned DataFrame index is timezone-aware in UTC.

        Args:
            asset: Asset instance
            timeframe: string timeframe as stored in OHLCV.timeframe (e.g. "M5", "H1")
            start_utc: naive or tz-aware datetime (interpreted as UTC if naive)
            end_utc: naive or tz-aware datetime (interpreted as UTC if naive)

        Returns:
            pd.DataFrame indexed by UTC-aware timestamps with columns ['open','high','low','close','volume']

        Raises:
            ValueError: if no rows are found for given range
        """
        logger.info("Loading OHLCV data for analysis", asset=asset.symbol, timeframe=timeframe)

        # Normalize start/end to UTC-aware datetimes
        def _ensure_utc(dt: datetime.datetime) -> datetime.datetime:
            if dt is None:
                return dt
            if dt.tzinfo is None:
                # assume provided naive datetimes are in UTC
                return dt.replace(tzinfo=datetime.timezone.utc)
            # convert to UTC
            return dt.astimezone(datetime.timezone.utc)

        start_utc = _ensure_utc(start_utc)
        end_utc = _ensure_utc(end_utc)

        # Query DB
        queryset = OHLCV.objects.filter(
            asset=asset, timeframe=timeframe, timestamp__range=(start_utc, end_utc)
        ).order_by("timestamp")

        if not queryset.exists():
            raise ValueError(f"No OHLCV data found for {asset.symbol} in the given range.")

        # Build DataFrame
        df = pd.DataFrame.from_records(
            queryset.values("timestamp", "open", "high", "low", "close", "volume")
        )

        # Ensure proper dtypes
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols + ["volume"]:
            if col in df.columns:
                # coerce to numeric (float for prices, int/float for volume)
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Parse timestamps and set index
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
        df.set_index("timestamp", inplace=True)

        # Guarantee index is tz-aware UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        # Reorder/keep only required columns
        df = df[["open", "high", "low", "close", "volume"]]

        return df


def ingest_ohlcv_data(
    symbol: str,
    timeframe: Timeframe,
    start_utc: datetime.datetime,
    end_utc: datetime.datetime,
) -> int:
    """
    Fetch OHLCV data from MT5Connector and persist into the OHLCV model.

    Args:
        symbol: market symbol (e.g. "EURUSD")
        timeframe: Timeframe enum value
        start_utc: start datetime (naive or tz-aware). If naive, treated as UTC.
        end_utc: end datetime (naive or tz-aware). If naive, treated as UTC.

    Returns:
        Number of persisted OHLCV rows.
    """
    logger.info("Starting OHLCV ingestion", symbol=symbol, timeframe=timeframe.value)

    # Ensure asset exists
    asset, _ = Asset.objects.get_or_create(symbol=symbol)

    # Normalize start/end to timezone-aware UTC for connector usage
    def _ensure_utc(dt: datetime.datetime) -> datetime.datetime:
        if dt is None:
            return dt
        if dt.tzinfo is None:
            return dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(datetime.timezone.utc)

    start_utc = _ensure_utc(start_utc)
    end_utc = _ensure_utc(end_utc)

    # Fetch dataframe from MT5 connector
    try:
        df: pd.DataFrame = MT5Connector.fetch_ohlcv(symbol, timeframe, start_utc, end_utc)
    except Exception:
        logger.exception("Failed to fetch OHLCV from MT5Connector", symbol=symbol, timeframe=timeframe.value)
        return 0

    if df is None or df.empty:
        logger.info("No OHLCV returned from connector", symbol=symbol, timeframe=timeframe.value)
        return 0

    # Ensure index is datetime and tz-aware in UTC
    if not isinstance(df.index, pd.DatetimeIndex):
        # attempt to parse a timestamp column or index
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
            df.set_index("timestamp", inplace=True)
        else:
            # try to convert index
            df.index = pd.to_datetime(df.index, utc=False)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Coerce numeric columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Prepare model instances for bulk insert
    ohlcv_records = []
    for ts, row in df.iterrows():
        try:
            # ensure timestamp is a Python datetime (naive or aware acceptable for DB)
            ts_py = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            record = OHLCV(
                asset=asset,
                timeframe=timeframe.value,
                timestamp=ts_py,
                open=row.get("open"),
                high=row.get("high"),
                low=row.get("low"),
                close=row.get("close"),
                volume=row.get("volume"),
            )
            ohlcv_records.append(record)
        except Exception:
            logger.exception("Failed to convert DataFrame row to OHLCV model", symbol=symbol, index=ts)

    if not ohlcv_records:
        logger.info("No valid OHLCV records prepared for insertion", symbol=symbol)
        return 0

    try:
        with transaction.atomic():
            created_objects = OHLCV.objects.bulk_create(ohlcv_records, ignore_conflicts=True)
    except Exception:
        logger.exception("Bulk create failed for OHLCV records", symbol=symbol)
        return 0

    ingested_count = len(created_objects) if created_objects is not None else 0
    logger.info(
        "OHLCV data ingestion complete",
        symbol=symbol,
        timeframe=timeframe.value,
        count=ingested_count,
    )
    return ingested_count

