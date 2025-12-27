# apps/analytics/management/commands/run_walk_forward.py
import yaml
from django.core.management.base import BaseCommand, CommandParser
from apps.analytics.backtesting.walk_forward import run_walk_forward_validation
from apps.analytics.models.train import load_training_data
from apps.market_data.models import Asset

class Command(BaseCommand):
    help = "Demonstrates the Walk-Forward Validation splitting logic."

    def add_arguments(self, parser: CommandParser):
        parser.add_argument("--config", type=str, required=True, help="Path to the base YAML config file.")

    def handle(self, *args, **options):
        with open(options["config"], 'r') as f:
            config = yaml.safe_load(f)
        
        self.stdout.write(self.style.SUCCESS("--- Walk-Forward Validation Engine ---"))
        self.stdout.write("NOTE: This command only demonstrates the data splitting logic for now.")
        
        train_cfg = config['training']
        asset = Asset.objects.get(symbol=train_cfg['asset_symbol'])
        
        # تحميل البيانات الكاملة
        X, prices = load_training_data(asset, train_cfg['timeframe'])
        full_df = X.join(prices)

        results = run_walk_forward_validation(full_df, config)

        self.stdout.write("\n--- Summary of Folds ---")
        for fold in results:
            self.stdout.write(
                f"Fold {fold['fold']}: Train [{fold['train_start']} -> {fold['train_end']}], "
                f"Test [{fold['test_start']} -> {fold['test_end']}]"
            )
        self.stdout.write(self.style.SUCCESS(f"\nDemonstration complete. {len(results)} folds were generated."))


        