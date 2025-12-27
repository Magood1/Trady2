import datetime
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PatternCandidateData:
    """
    A dataclass to represent a detected pattern candidate before persistence.
    This ensures type safety and immutability.
    """

    symbol: str
    timestamp: datetime.datetime
    pattern_type: str
    confidence: float
    meta: dict[str, Any]

    def __post_init__(self) -> None:
        """Validation after initialization."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.timestamp.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware.")

    def to_dict(self) -> dict[str, Any]:
        """Converts the dataclass to a dictionary."""
        return asdict(self)