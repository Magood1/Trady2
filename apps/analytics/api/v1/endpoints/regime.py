from django.utils import timezone
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.analytics.serializers import MarketRegimeSerializer
from apps.market_data.models import Asset, MarketRegime


class RegimeAPIView(APIView):
    """
    API endpoint to retrieve the latest market regime for a given symbol.
    """

    def get(self, request: Request, symbol: str) -> Response:
        """
        Returns the latest calculated market regime for a symbol.
        Can optionally specify a point in time with the 'at' query parameter.
        """
        at_str = request.query_params.get("at")
        at_timestamp = timezone.now()
        if at_str:
            try:
                at_timestamp = timezone.datetime.fromisoformat(at_str)
            except ValueError:
                return Response(
                    {"error": "Invalid 'at' timestamp format. Use ISO-8601."},
                    status=400,
                )

        try:
            asset = Asset.objects.get(symbol__iexact=symbol)
            latest_regime = (
                MarketRegime.objects.filter(asset=asset, timestamp__lte=at_timestamp)
                .order_by("-timestamp")
                .first()
            )
        except Asset.DoesNotExist:
            return Response(
                {"error": f"Asset with symbol '{symbol}' not found."}, status=404
            )

        if not latest_regime:
            return Response(
                {"error": "No regime data found for this asset."}, status=404
            )

        serializer = MarketRegimeSerializer(latest_regime)
        return Response(serializer.data)
