import datetime

import factory
from django.utils import timezone

from apps.market_data.models import Asset, OHLCV


class AssetFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = Asset
        django_get_or_create = ("symbol",)

    symbol = "EURUSD"
    asset_type = "FOREX"


class OHLCVFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = OHLCV

    asset = factory.SubFactory(AssetFactory)
    timeframe = "H1"
    timestamp = factory.LazyFunction(timezone.now)
    open = 1.1
    high = 1.11
    low = 1.09
    close = 1.105
    volume = 1000
