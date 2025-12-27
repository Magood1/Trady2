# apps/analytics/api/v1/endpoints/charts.py
import datetime
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from apps.analytics.services import ChartDataService

class CandlestickChartAPIView(APIView):
    """
    Provides data formatted for advanced candlestick charts, including
    OHLCV and all analytical annotations (regimes, verified patterns, signals).
    """
    # في المستقبل، قم بتأمين هذه النقطة
    # permission_classes = [permissions.IsAuthenticated]

    def get(self, request, symbol: str, timeframe: str):
        """
        Fetches chart data for a given symbol and timeframe.
        Query Parameters:
        - start_date (optional, ISO format, e.g., 2025-10-20T00:00:00Z)
        - end_date (optional, ISO format)
        - days (optional, integer, e.g., 7)
        """
        try:
            # تحديد النطاق الزمني بمرونة
            end_str = request.query_params.get('end_date')
            start_str = request.query_params.get('start_date')
            days_back = request.query_params.get('days')

            end_utc = timezone.now()
            if end_str:
                end_utc = datetime.datetime.fromisoformat(end_str.replace('Z', '+00:00'))

            start_utc = end_utc - datetime.timedelta(days=7) # الافتراضي هو آخر 7 أيام
            if start_str:
                start_utc = datetime.datetime.fromisoformat(start_str.replace('Z', '+00:00'))
            elif days_back:
                start_utc = end_utc - datetime.timedelta(days=int(days_back))
            
            # استدعاء الخدمة الجديدة لجلب كل البيانات
            chart_data = ChartDataService.get_chart_data(
                symbol=symbol.upper(),
                timeframe=timeframe.upper(),
                start_utc=start_utc,
                end_utc=end_utc
            )
            return Response(chart_data)
            
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.exception("Unexpected error in CandlestickChartAPIView")
            return Response(
                {"error": f"An unexpected error occurred: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        



        