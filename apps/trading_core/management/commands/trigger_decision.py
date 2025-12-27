from django.core.management.base import BaseCommand, CommandParser
from apps.market_data.tasks import trigger_decision_manager

class Command(BaseCommand):
    help = 'Triggers the decision manager for a given asset.'

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument('symbol', type=str)
        parser.add_argument('timeframe', type=str)

    def handle(self, *args, **options) -> None:
        symbol = options['symbol']
        timeframe = options['timeframe']
        self.stdout.write(f"Triggering decision manager for {symbol} {timeframe}...")

        # --- التعديل الحاسم هنا ---
        # نقوم ببناء قاموس يحاكي نتيجة المهمة السابقة
        mock_prev_result = {"symbol": symbol, "timeframe": timeframe}
        
        # نقوم بتمرير القاموس كوسيط واحد
        trigger_decision_manager.delay(mock_prev_result)
        # -------------------------

        self.stdout.write(self.style.SUCCESS("Task queued."))
        
# from django.core.management.base import BaseCommand
# from apps.market_data.tasks import trigger_decision_manager

# class Command(BaseCommand):
#     help = 'Triggers the decision manager for a given asset.'
#     def add_arguments(self, parser):
#         parser.add_argument('symbol', type=str)
#         parser.add_argument('timeframe', type=str)
#     def handle(self, *args, **options):
#         symbol = options['symbol']
#         timeframe = options['timeframe']
#         self.stdout.write(f"Triggering decision manager for {symbol} {timeframe}...")
#         trigger_decision_manager.delay(symbol, timeframe)
#         self.stdout.write(self.style.SUCCESS("Task queued."))
