# apps/analytics/management/commands/run_analysis.py
import pandas as pd
import structlog
from django.core.management.base import BaseCommand, CommandParser
from django.db import transaction
import yaml

from apps.analytics.features.pipeline import FeaturePipeline
from apps.analytics.services import OHLCVLoader
from apps.market_data.models import Asset
from apps.trading_core.models import FeatureVector

logger = structlog.get_logger(__name__)

class Command(BaseCommand):
    help = "Generates feature vectors for a specific asset, respecting the train/test split."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
        parser.add_argument("--rebuild", action="store_true", help="Delete all existing feature vectors before generating.")

    @transaction.atomic
    def handle(self, *args, **options) -> None:
        config_path = options["config"]
        rebuild = options["rebuild"]

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        asset_symbol = config['training']['asset_symbol']
        timeframe = config['training']['timeframe']
        split_date = pd.to_datetime(config['data_split']['split_date'], utc=True)

        asset = Asset.objects.get(symbol__iexact=asset_symbol)

        self.stdout.write(self.style.SUCCESS(f"--- Feature Generation for {asset.symbol} | Split Date: {split_date.date()} ---"))

        if rebuild:
            self.stdout.write(self.style.WARNING(f"Rebuild flag is set. Deleting ALL existing feature vectors for {asset.symbol}..."))
            deleted_count, _ = FeatureVector.objects.filter(asset=asset).delete()
            self.stdout.write(f"Successfully deleted {deleted_count} old feature vectors.")

        # تحميل كامل بيانات OHLCV مرة واحدة
        self.stdout.write("Loading all available OHLCV data...")
        loader = OHLCVLoader()
        all_ohlcv_df = loader.load_dataframe(
            asset=asset,
            timeframe=timeframe,
            start_utc=pd.Timestamp.min.tz_localize('UTC'),
            end_utc=pd.Timestamp.max.tz_localize('UTC')
        )
        self.stdout.write(f"Loaded {len(all_ohlcv_df)} total OHLCV records.")

        # إنشاء وحفظ الميزات لمجموعتي التدريب والاختبار
        train_df = all_ohlcv_df[all_ohlcv_df.index < split_date]
        test_df = all_ohlcv_df[all_ohlcv_df.index >= split_date]

        for period, ohlcv_data in [("TRAINING", train_df), ("TESTING", test_df)]:
            if ohlcv_data.empty:
                self.stdout.write(f"No data for {period} period. Skipping.")
                continue
            
            self.stdout.write(f"\n--- Building features for {period} period ({len(ohlcv_data)} records) ---")
            features_df = FeaturePipeline.build_feature_dataframe(asset.symbol, ohlcv_data)

            self.stdout.write("Saving feature vectors to the database...")
            new_feature_vectors = [
                FeatureVector(asset=asset, timestamp=ts, features=row)
                for ts, row in features_df.to_dict('index').items()
            ]
            FeatureVector.objects.bulk_create(new_feature_vectors, batch_size=500)
            self.stdout.write(self.style.SUCCESS(f"Saved {len(new_feature_vectors)} feature vectors for {period} period."))
            