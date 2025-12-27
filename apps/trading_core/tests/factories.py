# apps/trading_core/tests/factories.py
import factory
from django.utils import timezone
from decimal import Decimal
from apps.market_data.models import Asset
from apps.trading_core.models import TradingSignal

class AssetFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = Asset
        django_get_or_create = ('symbol',)
    
    symbol = factory.Sequence(lambda n: f'ASSET{n}')

class TradingSignalFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = TradingSignal

    asset = factory.SubFactory(AssetFactory)
    timestamp = factory.LazyFunction(timezone.now)
    signal_type = "BUY"
    entry_price = Decimal("1.1000")
    stop_loss = Decimal("1.0900")
    take_profit = Decimal("1.1100")
    position_size = Decimal("1000")
    status = "PENDING"


    