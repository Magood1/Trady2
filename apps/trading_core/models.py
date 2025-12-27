# apps/trading_core/models.py
"""
Models for pattern candidates, verifications, trading signals and feature vectors.

Contents
- PatternCandidate: raw pattern found by screeners (many candidates per asset)
- VerifiedPattern: result of a verifier (one-to-one with PatternCandidate)
- TradingSignal: a trading instruction derived from a verified pattern (one-to-one with PatternCandidate)
- FeatureVector: stored snapshot of engineered features for an asset at a timestamp (for ML / backtesting)
- Order: a record of a trade execution sent to a broker
- CircuitBreakerState: tracks the state of the system's risk management circuit breaker

Design notes
- VerifiedPattern uses OneToOneField with PatternCandidate to ensure a candidate can be verified at most once.
- TradingSignal.source_pattern is OneToOneField to enforce that a single PatternCandidate produces at most one TradingSignal.
- Order.signal is OneToOneField to ensure a signal results in at most one order.
- Confidence fields include validators to keep values within [0.0, 1.0].
- FeatureVector stores a JSON blob of features so different pipelines can ingest a canonical feature snapshot.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict

from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone

from apps.market_data.models import Asset


class PatternCandidate(models.Model):
    """
    مرشح نمط تم اكتشافه بواسطة الماسحات (screeners).

    يرتبط كل مرشح بأصل (Asset) ويتم ربطه بطابع زمني محدد لشمعة
    (إغلاق الشمعة الذي أدى إلى اكتشاف المرشح).
    """
    asset = models.ForeignKey(
        Asset,
        on_delete=models.CASCADE,
        related_name="pattern_candidates",
    )
    timestamp = models.DateTimeField(db_index=True, help_text="Candle close timestamp (UTC)")
    pattern_type = models.CharField(
        max_length=50,
        db_index=True,
        help_text="Pattern name, e.g., ENGULFING_BULLISH",
    )
    confidence = models.FloatField(
        help_text="Confidence score of the pattern [0.0, 1.0]",
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
    )
    meta = models.JSONField(default=dict, help_text="Algorithm parameters or other metadata")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("asset", "timestamp", "pattern_type")
        ordering = ["-timestamp"]
        verbose_name = "Pattern Candidate"
        verbose_name_plural = "Pattern Candidates"
        indexes = [
            models.Index(fields=["asset", "timestamp"]),
            models.Index(fields=["pattern_type"]),
        ]

    def __str__(self) -> str:
        ts_iso = self.timestamp.isoformat() if self.timestamp else "unknown-time"
        return f"{self.asset.symbol} @ {ts_iso} -> {self.pattern_type}"


class VerifiedPattern(models.Model):
    """
    نتيجة التحقق من PatternCandidate (على سبيل المثال عبر DTW).

    توجد علاقة واحد لواحد مع PatternCandidate بحيث يمكن أن يكون لكل مرشح
    سجل تحقق واحد على الأكثر.
    """
    candidate = models.OneToOneField(
        PatternCandidate,
        on_delete=models.CASCADE,
        related_name="verification",
    )
    verifier_type = models.CharField(
        max_length=20,
        db_index=True,
        help_text="Verifier identifier, e.g., DTW, LSTM",
    )
    confidence = models.FloatField(
        help_text="Confidence score from the verifier [0.0, 1.0]",
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
    )
    meta = models.JSONField(default=dict, help_text="Verifier metadata (e.g., dtw_distance)")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-candidate__timestamp"]
        verbose_name = "Verified Pattern"
        verbose_name_plural = "Verified Patterns"
        indexes = [
            models.Index(fields=["verifier_type"]),
            models.Index(fields=["confidence"]),
        ]

    def __str__(self) -> str:
        return f"{self.candidate.pattern_type} verified by {self.verifier_type} with score {self.confidence:.2f}"


class TradingSignal(models.Model):
    """
    إشارة تداول تم إنشاؤها من نمط تم التحقق منه (شراء/بيع)، يتم حفظها حتى يتمكن
    محرك التنفيذ أو المشغل من التصرف بناءً عليها.
    """
    SIGNAL_CHOICES = [
        ("BUY", "Buy"),
        ("SELL", "Sell"),
    ]

    STATUS_CHOICES = [
        ("PENDING", "Pending"),
        ("EXECUTED", "Executed"),
        ("CANCELLED", "Cancelled"),
    ]

    asset = models.ForeignKey(
        Asset,
        on_delete=models.CASCADE,
        related_name="trading_signals",
    )
    timestamp = models.DateTimeField(db_index=True, help_text="Timestamp of the signal generation")

    source_pattern = models.OneToOneField(
        PatternCandidate,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="generated_signal",
        help_text="The PatternCandidate that produced this signal (unique constraint via OneToOneField).",
    )

    signal_type = models.CharField(max_length=4, choices=SIGNAL_CHOICES)
    entry_price = models.DecimalField(max_digits=18, decimal_places=8)
    stop_loss = models.DecimalField(max_digits=18, decimal_places=8)
    take_profit = models.DecimalField(max_digits=18, decimal_places=8)
    position_size = models.DecimalField(max_digits=20, decimal_places=8)

    status = models.CharField(
        max_length=10,
        choices=STATUS_CHOICES,
        default="PENDING",
        db_index=True,
        help_text="Signal lifecycle state",
    )

    meta = models.JSONField(default=dict, help_text="Supporting data like regime, confidence, indicators, etc.")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-timestamp"]
        verbose_name = "Trading Signal"
        verbose_name_plural = "Trading Signals"
        indexes = [
            models.Index(fields=["asset", "status"]),
            models.Index(fields=["timestamp"]),
        ]

    def __str__(self) -> str:
        symbol = getattr(self.asset, "symbol", "unknown")
        return f"{self.signal_type} {symbol} @ {self.entry_price}"

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of the signal."""
        return {
            "id": self.pk,
            "asset": getattr(self.asset, "symbol", None),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "signal_type": self.signal_type,
            "entry_price": str(self.entry_price),
            "stop_loss": str(self.stop_loss),
            "take_profit": str(self.take_profit),
            "position_size": str(self.position_size),
            "status": self.status,
            "meta": self.meta,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "source_pattern_id": self.source_pattern.pk if self.source_pattern else None,
        }


class FeatureVector(models.Model):
    """
    يخزن لقطة من الميزات المهندسة لأصل في طابع زمني محدد.

    يحتوي حقل `features` من نوع JSON على قاموس يربط اسم الميزة بقيمتها،
    على سبيل المثال: {"rsi": 42.3, "atr": 0.0012, "sma_20": 1.2345, ...}
    """
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE, related_name="feature_vectors")
    timestamp = models.DateTimeField(db_index=True, help_text="The timestamp of the candle (UTC)")
    features = models.JSONField(default=dict, help_text="A dictionary of all feature values")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("asset", "timestamp")
        ordering = ["-timestamp"]
        verbose_name = "Feature Vector"
        verbose_name_plural = "Feature Vectors"
        indexes = [
            models.Index(fields=["asset", "timestamp"]),
        ]

    def __str__(self) -> str:
        ts = self.timestamp.isoformat() if self.timestamp else "unknown-time"
        return f"Features for {self.asset.symbol} @ {ts}"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of the feature vector."""
        return {
            "id": self.pk,
            "asset": getattr(self.asset, "symbol", None),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "features": self.features,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# -----------------------------------------------
# New: Execution and Risk Management Models
# -----------------------------------------------

class Order(models.Model):
    """
    سجل لجميع أوامر التداول المرسلة إلى الوسيط.
    يوفر مسار تدقيق كامل.
    """
    STATUS_CHOICES = [
        ("SENT", "Sent"),
        ("FILLED", "Filled"),
        ("CANCELLED", "Cancelled"),
        ("REJECTED", "Rejected"),
        ("FAILED", "Failed"),
    ]

    signal = models.OneToOneField(TradingSignal, on_delete=models.CASCADE, related_name="order")
    broker_order_id = models.CharField(max_length=100, null=True, blank=True, db_index=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default="SENT")
    sent_at = models.DateTimeField(auto_now_add=True)
    executed_at = models.DateTimeField(null=True, blank=True)
    broker_response = models.JSONField(default=dict, help_text="The raw response from the broker API.")
    
    class Meta:
        ordering = ["-sent_at"]
        verbose_name = "Execution Order"
        verbose_name_plural = "Execution Orders"

    def __str__(self):
        return f"Order for Signal {self.signal.id} - Status: {self.status}"


class CircuitBreakerState(models.Model):
    """
    يخزن الحالة الحالية لقاطع الدائرة (Circuit Breaker).
    """
    is_tripped = models.BooleanField(default=False, db_index=True)
    tripped_at = models.DateTimeField(null=True, blank=True)
    reason = models.CharField(max_length=255, null=True, blank=True)
    consecutive_losses = models.PositiveIntegerField(default=0)
    last_reset_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Circuit Breaker: {'TRIPPED' if self.is_tripped else 'OK'}"