# apps/analytics/management/commands/seed_data_from_json.py
import json
from decimal import Decimal
from django.core.management.base import BaseCommand, CommandParser
from django.db import transaction
from django.utils.dateparse import parse_datetime
from apps.market_data.models import Asset, OHLCV
from apps.trading_core.models import FeatureVector

class Command(BaseCommand):
    help = "Seeds the database with OHLCV and FeatureVector data from a provided JSON file."

    def add_arguments(self, parser: CommandParser):
        parser.add_argument("json_file", type=str, help="Path to the JSON data file.")

    @transaction.atomic
    def handle(self, *args, **options):
        file_path = options["json_file"]
        self.stdout.write(f"Loading data from {file_path}...")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        symbol = data.get("symbol")
        timeframe = data.get("timeframe")
        ohlcv_data = data.get("ohlcv", [])

        if not all([symbol, timeframe, ohlcv_data]):
            self.stderr.write(self.style.ERROR("JSON file is missing symbol, timeframe, or ohlcv data."))
            return

        asset, created = Asset.objects.get_or_create(symbol=symbol)
        if created:
            self.stdout.write(self.style.SUCCESS(f"Created new asset: {symbol}"))

        ohlcv_records = []
        feature_records = []
        for row in ohlcv_data:
            ts = parse_datetime(row['time'])
            ohlcv_records.append(
                OHLCV(
                    asset=asset,
                    timeframe=timeframe,
                    timestamp=ts,
                    open=Decimal(row['open']),
                    high=Decimal(row['high']),
                    low=Decimal(row['low']),
                    close=Decimal(row['close']),
                    volume=int(row['volume'])
                )
            )
            # Create a simplified FeatureVector from OHLCV for seeding purposes
            feature_records.append(
                FeatureVector(
                    asset=asset,
                    timestamp=ts,
                    features={
                        "open": float(row['open']),
                        "high": float(row['high']),
                        "low": float(row['low']),
                        # NOTE: "close" is intentionally omitted here to prevent the overlap error
                    }
                )
            )
        
        OHLCV.objects.bulk_create(ohlcv_records, ignore_conflicts=True)
        self.stdout.write(self.style.SUCCESS(f"Seeded {len(ohlcv_records)} OHLCV records."))
        
        FeatureVector.objects.bulk_create(feature_records, ignore_conflicts=True)
        self.stdout.write(self.style.SUCCESS(f"Seeded {len(feature_records)} basic FeatureVector records."))