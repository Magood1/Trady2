# apps/analytics/management/commands/train_model.py
import yaml
import structlog
from django.core.management.base import BaseCommand, CommandParser

# Updated import: We only need run_training_pipeline now, as it encapsulates the whole logic
from apps.analytics.models.train import run_training_pipeline

logger = structlog.get_logger(__name__)

class Command(BaseCommand):
    help = "Trains the Meta-Model using the Production Pipeline (Triple Barrier + Survivor Features)."

    def add_arguments(self, parser: CommandParser):
        parser.add_argument(
            "--config",
            type=str,
            required=True,
            help="Path to the YAML configuration file for training.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for reproducibility.",
        )

    def handle(self, *args, **options):
        config_path = options["config"]
        seed = options["seed"]

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Failed to load or parse config file: {e}"))
            return
        
        # Override seed in config if provided via CLI
        if 'training' not in config:
            config['training'] = {}
        config['training']['seed'] = seed

        self.stdout.write(self.style.SUCCESS(f"--- Starting Production Training Pipeline ---"))
        self.stdout.write(f"Asset: {config['training'].get('asset_symbol')}")
        self.stdout.write(f"Timeframe: {config['training'].get('timeframe')}")
        self.stdout.write(f"Config: {config_path}")

        try:
            # Delegate to the core logic in models/train.py
            # This ensures consistency between what is defined in the model logic and what is executed.
            run_training_pipeline(config, seed)
            
            self.stdout.write(self.style.SUCCESS("\nTraining completed successfully."))
            self.stdout.write("Next steps:")
            self.stdout.write("1. Go to Django Admin -> Model Registry.")
            self.stdout.write("2. Activate the newly created model.")
            self.stdout.write("3. Monitor the logs for 'Meta-Model ACCEPTED' signals.")

        except Exception as e:
            logger.exception("Training pipeline failed.")
            self.stderr.write(self.style.ERROR(f"Training failed: {e}"))