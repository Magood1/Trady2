# apps/market_data/management/commands/ingest_historical_data.py
from django.core.management.base import BaseCommand, CommandParser
from apps.market_data.tasks import ingest_historical_data_task

class Command(BaseCommand):
    help = "Ingests historical OHLCV data for a given symbol and timeframe."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("symbol", type=str, help="The trading symbol (e.g., EURUSD)")
        parser.add_argument("timeframe", type=str, help="The timeframe (e.g., M15, H1)")
        parser.add_argument(
            "--days",
            type=int,
            default=3650,
            help="Number of days back to ingest data for. Default is 365.",
        )

    def handle(self, *args, **options) -> None:
        symbol = options["symbol"]
        timeframe = options["timeframe"]
        days_back = options["days"]
        
        self.stdout.write(
            f"Starting ingestion task for {symbol} on {timeframe} timeframe for the last {days_back} days..."
        )
        task = ingest_historical_data_task.delay(symbol, timeframe, days_back)
        self.stdout.write(self.style.SUCCESS(f"Task {task.id} successfully queued."))

        