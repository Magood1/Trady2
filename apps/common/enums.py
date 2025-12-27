from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import MetaTrader5 as mt5


class Timeframe(Enum):
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"

    def to_mt5(self) -> int:
        import MetaTrader5 as mt5

        return getattr(mt5, f"TIMEFRAME_{self.value}")
