from datetime import datetime
from typing import Any

from rest_framework import serializers

from apps.common.enums import Timeframe
from apps.market_data.models import OHLCV


class OHLCVSerializer(serializers.ModelSerializer):
    """
    Serializer for OHLCV model.
    """
    class Meta:
        model = OHLCV
        fields = "__all__"


class OHLCVIngestRequestSerializer(serializers.Serializer):
    """
    Serializer for OHLCV ingestion request.
    """
    symbol = serializers.CharField(max_length=20)
    timeframe = serializers.ChoiceField(choices=[(tf.value, tf.value) for tf in Timeframe])
    start_date = serializers.DateTimeField()
    end_date = serializers.DateTimeField()

    def validate_start_date(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise serializers.ValidationError("start_date must be timezone aware (UTC).")
        return value

    def validate_end_date(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise serializers.ValidationError("end_date must be timezone aware (UTC).")
        return value

    def validate(self, data: dict[str, Any]) -> dict[str, Any]:
        if data["start_date"] >= data["end_date"]:
            raise serializers.ValidationError("start_date must be before end_date.")
        return data