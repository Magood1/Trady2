from rest_framework import serializers

from apps.market_data.models import MarketRegime, OHLCV
from apps.trading_core.models import PatternCandidate


class MarketRegimeSerializer(serializers.ModelSerializer):
    """Serializer for the MarketRegime model."""
    symbol = serializers.CharField(source="asset.symbol", read_only=True)
    class Meta:
        model = MarketRegime
        fields = ["symbol", "timestamp", "regime", "confidence", "meta"]


class PatternCandidateSerializer(serializers.ModelSerializer):
    """Serializer for the PatternCandidate model."""
    symbol = serializers.CharField(source="asset.symbol", read_only=True)
    class Meta:
        model = PatternCandidate
        fields = ["id", "symbol", "timestamp", "pattern_type", "confidence", "meta"]


# --- Serializers for Charting Endpoint ---

class OHLCVChartSerializer(serializers.ModelSerializer):
    """Optimized serializer for candlestick chart OHLCV data."""
    time = serializers.DateTimeField(source='timestamp')
    class Meta:
        model = OHLCV
        fields = ['time', 'open', 'high', 'low', 'close', 'volume']


class AnnotationSerializer(serializers.Serializer):
    """Serializer for chart annotations (patterns, regimes, etc.)."""
    time = serializers.DateTimeField()
    text = serializers.CharField()
    color = serializers.CharField(required=False)
    position = serializers.CharField(default='aboveBar')
    shape = serializers.CharField(default='arrowDown')


