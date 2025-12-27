import django_filters
from apps.market_data.models import MarketRegime
from apps.trading_core.models import PatternCandidate


class PatternCandidateFilter(django_filters.FilterSet):
    """Advanced filters for the PatternCandidate model."""
    
    start_date = django_filters.DateTimeFilter(field_name="timestamp", lookup_expr='gte')
    end_date = django_filters.DateTimeFilter(field_name="timestamp", lookup_expr='lte')
    
    class Meta:
        model = PatternCandidate
        fields = {
            'asset__symbol': ['exact', 'in'],
            'pattern_type': ['exact', 'in'],
            'confidence': ['gte', 'lte'],
        }


class MarketRegimeFilter(django_filters.FilterSet):
    """Advanced filters for the MarketRegime model."""

    start_date = django_filters.DateTimeFilter(field_name="timestamp", lookup_expr='gte')
    end_date = django_filters.DateTimeFilter(field_name="timestamp", lookup_expr='lte')

    class Meta:
        model = MarketRegime
        fields = {
            'asset__symbol': ['exact', 'in'],
            'regime': ['exact', 'in'],
        }