from django.core.management.base import BaseCommand
from django.db import transaction
from apps.trading_core.models import TradingSignal

class Command(BaseCommand):
    help = 'De-duplicates TradingSignal objects, keeping the latest one for each source_pattern.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("--- Starting Signal De-duplication Process ---"))

        signals_to_check = TradingSignal.objects.exclude(
            source_pattern__isnull=True
        ).order_by('source_pattern_id', '-timestamp')

        if not signals_to_check.exists():
            self.stdout.write(self.style.SUCCESS("No signals with source patterns to de-duplicate. Exiting."))
            return

        seen_patterns = set()
        ids_to_delete = []

        for signal in signals_to_check:
            if signal.source_pattern_id not in seen_patterns:
                seen_patterns.add(signal.source_pattern_id)
            else:
                ids_to_delete.append(signal.id)

        if not ids_to_delete:
            self.stdout.write(self.style.SUCCESS("No duplicate signals found. Database is clean."))
            return

        self.stdout.write(f"Found {len(ids_to_delete)} duplicate signals to delete.")

        try:
            with transaction.atomic():
                deleted_count, _ = TradingSignal.objects.filter(id__in=ids_to_delete).delete()
            self.stdout.write(self.style.SUCCESS(f"Successfully deleted {deleted_count} duplicate signals."))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"An error occurred during deletion: {e}"))

        self.stdout.write(self.style.SUCCESS("--- De-duplication Process Finished ---"))
        
# from django.core.management.base import BaseCommand
# from django.db import transaction
# from apps.trading_core.models import TradingSignal

# class Command(BaseCommand):
#     help = 'De-duplicates TradingSignal objects, keeping the latest one for each source_pattern.'

#     def handle(self, *args, **options):
#         self.stdout.write("Starting de-duplication of TradingSignals...")

#         signals_to_check = TradingSignal.objects.exclude(
#             source_pattern__isnull=True
#         ).order_by('source_pattern_id', '-timestamp')

#         if not signals_to_check.exists():
#             self.stdout.write(self.style.SUCCESS("No signals with source patterns to de-duplicate."))
#             return

#         seen_patterns = set()
#         ids_to_keep = []
#         ids_to_delete = []

#         for signal in signals_to_check:
#             if signal.source_pattern_id not in seen_patterns:
#                 ids_to_keep.append(signal.id)
#                 seen_patterns.add(signal.source_pattern_id)
#             else:
#                 ids_to_delete.append(signal.id)

#         if not ids_to_delete:
#             self.stdout.write(self.style.SUCCESS("No duplicate signals found."))
#             return

#         self.stdout.write(f"Found {len(ids_to_delete)} duplicate signals to delete.")

#         with transaction.atomic():
#             deleted_count, _ = TradingSignal.objects.filter(id__in=ids_to_delete).delete()

#         self.stdout.write(self.style.SUCCESS(f"Successfully deleted {deleted_count} duplicate signals."))