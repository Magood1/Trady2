#apps/market_data/connectors/mt5_connector.py
import datetime
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd
import structlog
from django.conf import settings
from tenacity import retry, stop_after_attempt, wait_exponential

from apps.common.enums import Timeframe

logger = structlog.get_logger(__name__)

class MT5Connector:
    """
    A robust, class-based connector for MetaTrader 5 with managed state.
    """
    _initialized = False

    @classmethod
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
    def initialize(cls) -> None:
        """
        Initializes the MT5 connection if not already active.
        This method is safe to call multiple times.
        """
        if cls._initialized and mt5.terminal_info() is not None:
            return

        login_details = {}
        if not settings.MT5_LOCAL_TERMINAL:
            if not all([settings.MT5_LOGIN, settings.MT5_PASSWORD, settings.MT5_SERVER]):
                logger.error("Remote MT5 connection details are missing in .env file.")
                raise ConnectionError("Remote MT5 connection details are missing.")
            login_details = {
                "login": int(settings.MT5_LOGIN),
                "password": settings.MT5_PASSWORD,
                "server": settings.MT5_SERVER,
            }

        if not mt5.initialize(**login_details):
            error = mt5.last_error()
            logger.error("MT5 initialize() failed", error_code=error, **login_details)
            raise ConnectionError(f"Failed to initialize MT5: {error}")

        cls._initialized = True
        logger.info("MT5 connection initialized successfully.")

    @classmethod
    def shutdown(cls) -> None:
        """Shuts down the MT5 connection."""
        if cls._initialized:
            mt5.shutdown()
            cls._initialized = False
            logger.info("MT5 connection shut down.")

    @classmethod
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
    def fetch_ohlcv(
        cls, symbol: str, timeframe: Timeframe, start_utc: datetime.datetime, end_utc: datetime.datetime
    ) -> pd.DataFrame:
        """Fetches OHLCV data, ensuring the connection is initialized."""
        cls.initialize()

        rates = mt5.copy_rates_range(symbol, timeframe.to_mt5(), start_utc, end_utc)

        if rates is None:
            error = mt5.last_error()
            logger.error("Failed to fetch rates from MT5", symbol=symbol, error=error)
            return pd.DataFrame()

        if len(rates) == 0:
            logger.warning("No data returned for the given range", symbol=symbol, start=start_utc, end=end_utc)
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.rename(
            columns={"time": "timestamp", "open": "open", "high": "high", "low": "low", "close": "close", "tick_volume": "volume"},
            inplace=True,
        )
        df.drop(columns=['spread', 'real_volume'], inplace=True, errors='ignore')
        df.set_index('timestamp', inplace=True)

        logger.info("Successfully fetched OHLCV data", symbol=symbol, rows=len(df))
        return df


# import datetime
# from typing import Optional

# import MetaTrader5 as mt5
# import pandas as pd
# import structlog
# from django.conf import settings
# from tenacity import retry, stop_after_attempt, wait_exponential

# from apps.common.enums import Timeframe

# logger = structlog.get_logger(__name__)


# class MT5Connector:
#     _initialized = False

#     @classmethod
#     @retry(
#         wait=wait_exponential(multiplier=1, min=2, max=10),
#         stop=stop_after_attempt(3),
#         reraise=True,
#     )
#     def initialize(cls) -> None:
#         if cls._initialized:
#             return

#         if settings.MT5_LOCAL_TERMINAL:
#             if not mt5.initialize():
#                 raise ConnectionError("Failed to initialize local MT5 terminal")
#         else:
#             if not mt5.initialize(
#                 login=settings.MT5_LOGIN,
#                 password=settings.MT5_PASSWORD,
#                 server=settings.MT5_SERVER,
#             ):
#                 raise ConnectionError(
#                     f"Failed to initialize remote MT5: {mt5.last_error()}"
#                 )

#         cls._initialized = True
#         logger.info("MT5 connection initialized successfully.")

#     @classmethod
#     def shutdown(cls) -> None:
#         if cls._initialized:
#             mt5.shutdown()
#             cls._initialized = False
#             logger.info("MT5 connection shut down.")

#     @classmethod
#     @retry(
#         wait=wait_exponential(multiplier=1, min=2, max=10),
#         stop=stop_after_attempt(3),
#         reraise=True,
#     )
#     def fetch_ohlcv(
#         cls,
#         symbol: str,
#         timeframe: Timeframe,
#         start_utc: datetime.datetime,
#         end_utc: datetime.datetime,
#     ) -> pd.DataFrame:
#         cls.initialize()

#         rates = mt5.copy_rates_range(symbol, timeframe.to_mt5(), start_utc, end_utc)

#         if rates is None:
#             logger.error(
#                 "Failed to fetch rates from MT5", symbol=symbol, error=mt5.last_error()
#             )
#             return pd.DataFrame()

#         if len(rates) == 0:
#             logger.warning(
#                 "No data returned for the given range",
#                 symbol=symbol,
#                 start=start_utc,
#                 end=end_utc,
#             )
#             return pd.DataFrame()

#         df = pd.DataFrame(rates)
#         df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
#         df.rename(
#             columns={
#                 "time": "timestamp",
#                 "open": "open",
#                 "high": "high",
#                 "low": "low",
#                 "close": "close",
#                 "tick_volume": "volume",
#             },
#             inplace=True,
#         )
#         df.drop(columns=["spread", "real_volume"], inplace=True)
#         df.set_index("timestamp", inplace=True)

#         logger.info("Successfully fetched OHLCV data", symbol=symbol, rows=len(df))
#         return df
