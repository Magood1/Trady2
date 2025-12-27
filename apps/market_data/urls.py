from django.urls import path

from .views import OHLCVIngestAPIView

# هذه القائمة هي ما يبحث عنه Django
urlpatterns = [
    path("ohlcv/ingest/", OHLCVIngestAPIView.as_view(), name="ohlcv_ingest"),
]