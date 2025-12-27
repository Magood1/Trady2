from django.urls import path, include
from apps.analytics.api.v1.endpoints.regime import RegimeAPIView
from .routers import router
from .endpoints.charts import CandlestickChartAPIView
from .endpoints.dashboard import DashboardSummaryAPIView
from .endpoints.predict import PredictAPIView 

# --[ إضافة جديدة: استيراد العرض الجديد للرسم البياني ]--
from apps.analytics.api.v1.endpoints.charts import CandlestickChartAPIView


urlpatterns = [
    # URLs for ModelViewSets are handled by the router
    path('', include(router.urls)),

    path("regime/<str:symbol>/", RegimeAPIView.as_view(), name="get_regime"),
    
    path("predict/", PredictAPIView.as_view(), name="predict"),

    # Custom APIViews
    path(
        'charts/candlestick/<str:symbol>/<str:timeframe>/', 
        CandlestickChartAPIView.as_view(), 
        name='candlestick_chart'
    ),
    path(
        'dashboard/summary/',
        DashboardSummaryAPIView.as_view(),
        name='dashboard_summary'
    ),

    # --[ إضافة جديدة: نقطة النهاية المتقدمة للرسم البياني ]--
    path(
        'charts/candlestick/<str:symbol>/<str:timeframe>/', 
        CandlestickChartAPIView.as_view(), 
        name='candlestick_chart'
    ),

]


