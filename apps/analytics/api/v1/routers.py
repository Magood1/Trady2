from rest_framework.routers import DefaultRouter
from rest_framework import viewsets
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import permissions, filters

from apps.trading_core.models import PatternCandidate
from apps.market_data.models import MarketRegime
from apps.analytics.serializers import PatternCandidateSerializer, MarketRegimeSerializer
from apps.analytics.filters import PatternCandidateFilter, MarketRegimeFilter

class MarketRegimeViewSet(viewsets.ReadOnlyModelViewSet):
    """
    API endpoint for viewing Market Regimes.
    Supports filtering by symbol, regime, and date range.
    """
    queryset = MarketRegime.objects.all().order_by('-timestamp')
    serializer_class = MarketRegimeSerializer
    #permission_classes = [permissions.IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_class = MarketRegimeFilter
    ordering_fields = ['timestamp', 'confidence']


class PatternCandidateViewSet(viewsets.ReadOnlyModelViewSet):
    """
    API endpoint for viewing Pattern Candidates.
    Supports advanced filtering and ordering.
    """
    queryset = PatternCandidate.objects.all().order_by('-timestamp')
    serializer_class = PatternCandidateSerializer
    #permission_classes = [permissions.IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_class = PatternCandidateFilter
    ordering_fields = ['timestamp', 'confidence']
    

# Create a router and register our viewsets with it.
router = DefaultRouter()
router.register(r'regimes', MarketRegimeViewSet)
router.register(r'patterns', PatternCandidateViewSet)