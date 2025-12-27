from django.db import models


class Asset(models.Model):
    symbol = models.CharField(
        max_length=20, unique=True, db_index=True, help_text="e.g., EURUSD"
    )
    asset_type = models.CharField(
        max_length=10,
        choices=[("FOREX", "Forex"), ("CRYPTO", "Crypto")],
        default="FOREX",
    )
    description = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return self.symbol


class OHLCV(models.Model):
    asset = models.ForeignKey(
        Asset, on_delete=models.CASCADE, related_name="ohlcv_data"
    )
    timeframe = models.CharField(max_length=5, help_text="e.g., M1, H1, D1")
    timestamp = models.DateTimeField(
        db_index=True, help_text="Candle open timestamp (UTC)"
    )
    open = models.DecimalField(max_digits=18, decimal_places=8)
    high = models.DecimalField(max_digits=18, decimal_places=8)
    low = models.DecimalField(max_digits=18, decimal_places=8)
    close = models.DecimalField(max_digits=18, decimal_places=8)
    volume = models.BigIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("asset", "timeframe", "timestamp")
        ordering = ["-timestamp"]

    def __str__(self) -> str:
        return f"{self.asset.symbol} ({self.timeframe}) @ {self.timestamp}"


from django.db import models

# ... (نموذج Asset و OHLCV من Sprint 0) ...


class MarketRegime(models.Model):
    """
    Stores the calculated market regime for an asset at a specific point in time.
    """

    asset = models.ForeignKey(Asset, on_delete=models.CASCADE, related_name="regimes")
    timestamp = models.DateTimeField(
        db_index=True, help_text="Timestamp of the analysis (UTC)"
    )
    regime = models.CharField(
        max_length=20,
        choices=[
            ("TRENDING", "Trending"),
            ("MEAN_REVERTING", "Mean-Reverting"),
            ("HIGH_VOLATILITY", "High Volatility"),
            ("RANDOM", "Random/Noise"),
        ],
    )
    confidence = models.FloatField(
        default=0.0, help_text="Confidence score of the classification [0.0, 1.0]"
    )
    meta = models.JSONField(
        default=dict, help_text="Metadata, e.g., Hurst value, ATR value"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("asset", "timestamp")
        ordering = ["-timestamp"]
        verbose_name = "Market Regime"
        verbose_name_plural = "Market Regimes"

    def __str__(self) -> str:
        return f"{self.asset.symbol} @ {self.timestamp.isoformat()} -> {self.regime}"

  
    
   
        # hurst_value = models.FloatField(null=True)
        # atr_value = models.FloatField(null=True)
   