import datetime
from collections import Counter
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import permissions
from apps.trading_core.models import TradingSignal, PatternCandidate
from apps.market_data.models import MarketRegime, Asset

class DashboardSummaryAPIView(APIView):
    """
    Provides a high-level summary of the system's state for a dashboard.
    This version is compatible with SQLite and other database backends.
    """
    #permission_classes = [permissions.IsAuthenticated] # ملاحظة: ستحتاج إلى مصادقة للوصول
    
    def get(self, request):
        today = timezone.now().date()
        
        signals_today = TradingSignal.objects.filter(created_at__date=today).count()
        
        latest_patterns = PatternCandidate.objects.order_by('-timestamp')[:5]
        
        # --- بداية الكود المعدل ---
        # الحصول على أحدث نظام سوق لكل أصل بطريقة متوافقة مع جميع قواعد البيانات
        
        assets = Asset.objects.all()
        latest_regimes = []
        for asset in assets:
            latest_regime_for_asset = MarketRegime.objects.filter(asset=asset).order_by('-timestamp').first()
            if latest_regime_for_asset:
                latest_regimes.append(latest_regime_for_asset.regime)
        
        # حساب توزيع الأنظمة يدويًا
        regime_counts = Counter(latest_regimes)
        regime_distribution = [{"regime": regime, "count": count} for regime, count in regime_counts.items()]
        # --- نهاية الكود المعدل ---

        summary = {
            "signals_generated_today": signals_today,
            "latest_patterns": [
                f"{p.asset.symbol} @ {p.timestamp.strftime('%Y-%m-%d %H:%M')}: {p.pattern_type}"
                for p in latest_patterns
            ],
            "regime_distribution": regime_distribution
        }
        
        return Response(summary)