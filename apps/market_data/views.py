import structlog
from django.db import connection
from django.db.utils import OperationalError
from project_trady2.celery import app as celery_app
from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .serializers import OHLCVIngestRequestSerializer
from .tasks import ingest_historical_data_task

logger = structlog.get_logger(__name__)


class HealthCheckAPIView(APIView):
    def get(self, request: Request) -> Response:
        try:
            connection.ensure_connection()
            db_ok = True
        except OperationalError:
            db_ok = False

        try:
            celery_app.control.ping(timeout=1)
            redis_ok = True
        except Exception:
            redis_ok = False

        if db_ok and redis_ok:
            return Response({"status": "ok", "db": "ok", "redis": "ok"})

        return Response(
            {
                "status": "error",
                "db": "ok" if db_ok else "error",
                "redis": "ok" if redis_ok else "error",
            },
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


class OHLCVIngestAPIView(APIView):
    def post(self, request: Request) -> Response:
        serializer = OHLCVIngestRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        days_back = (data["end_date"] - data["start_date"]).days

        task = ingest_historical_data_task.delay(
            data["symbol"], data["timeframe"], days_back
        )

        return Response(
            {"message": "OHLCV ingestion task started.", "task_id": task.id},
            status=status.HTTP_202_ACCEPTED,
        )
