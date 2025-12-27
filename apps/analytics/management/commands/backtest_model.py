# apps/analytics/management/commands/backtest_model.py
import yaml
import pandas as pd
import structlog
from django.core.management.base import BaseCommand, CommandParser

from apps.mlops.services import get_model_by_version
from apps.analytics.models.train import load_training_data
from apps.analytics.backtesting.run import run_vectorized_backtest, generate_report
from apps.market_data.models import Asset, OHLCV

logger = structlog.get_logger(__name__)

class Command(BaseCommand):
    help = "Runs an out-of-sample backtest using specialized models with logical filters."

    def add_arguments(self, parser: CommandParser):
        parser.add_argument("--long-model-version", type=str, required=True, help="Version of the LONG-only model.")
        parser.add_argument("--short-model-version", type=str, required=True, help="Version of the SHORT-only model.")
        parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
        parser.add_argument("--output", type=str, default="backtest_report", help="Prefix for report files.")

    def handle(self, *args, **options):
        config_path = options["config"]
        output_prefix = options["output"]

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        split_date = pd.to_datetime(config['data_split']['split_date'], utc=True)

        self.stdout.write(f"Fetching LONG model: {options['long_model_version']}")
        long_model_info = get_model_by_version(options["long_model_version"])
        
        self.stdout.write(f"Fetching SHORT model: {options['short_model_version']}")
        short_model_info = get_model_by_version(options["short_model_version"])

        if not long_model_info or not short_model_info:
            self.stderr.write(self.style.ERROR("One or both model versions not found."))
            return
        
        long_model, long_registry = long_model_info
        short_model, short_registry = short_model_info
        
        self.stdout.write(self.style.SUCCESS(f"\n--- Running OUT-OF-SAMPLE Backtest on data from {split_date.date()} onwards ---"))

        train_cfg = config['training']
        asset = Asset.objects.get(symbol=train_cfg['asset_symbol'])

        self.stdout.write("Loading all OHLCV data for backtest period...")
        all_ohlcv = OHLCV.objects.filter(asset=asset, timeframe=train_cfg['timeframe']).order_by('timestamp')
        prices_df = pd.DataFrame.from_records(all_ohlcv.values('timestamp', 'open', 'high', 'low', 'close'))
        prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], utc=True)
        prices_df = prices_df.set_index('timestamp')

        # --- **بداية منطقة الإصلاح الحرجة** ---
        # تحويل أنواع البيانات من Decimal (القادمة من قاعدة البيانات) إلى float (المستخدمة في الحسابات).
        # هذا يحل خطأ TypeError بشكل مباشر من المصدر.
        for col in ['open', 'high', 'low', 'close']:
            prices_df[col] = pd.to_numeric(prices_df[col], errors='coerce')
        # --- **نهاية منطقة الإصلاح الحرجة** ---

        self.stdout.write("Loading feature data...")
        X, _ = load_training_data(asset, train_cfg['timeframe'])
        
        # **تطبيق صارم للتقسيم الزمني**
        X_test = X[X.index >= split_date]
        prices_test = prices_df[prices_df.index >= split_date]

        if X_test.empty:
            self.stderr.write(self.style.ERROR("No out-of-sample data found for backtesting after the split date."))
            return
            
        all_features = set(long_registry.feature_list) | set(short_registry.feature_list) | {'momentum_filter_200', 'rsi'}
        missing = [f for f in all_features if f not in X_test.columns]
        if missing:
             self.stderr.write(self.style.ERROR(f"Feature mismatch! Missing from test data: {missing}"))
             return
        
        self.stdout.write("Generating predictions from specialized models...")
        long_probs = long_model.predict_proba(X_test[long_registry.feature_list].values)[:, 1]
        short_probs = short_model.predict_proba(X_test[short_registry.feature_list].values)[:, 1]
        
        signals_df = pd.DataFrame({'long_signal_raw': long_probs, 'short_signal_raw': short_probs}, index=X_test.index)
        signals_df = signals_df.join(X_test[['momentum_filter_200', 'rsi']], how='inner')

        self.stdout.write("Applying strategic filters to raw predictions...")
        bt_cfg = config.get('backtesting', {})
        threshold_long = bt_cfg.get('threshold_long', 0.5)
        threshold_short = bt_cfg.get('threshold_short', 0.5)
        
        long_condition = (
            (signals_df['long_signal_raw'] > threshold_long) &
            (signals_df['momentum_filter_200'] == 1.0) &
            (signals_df['rsi'] < 60)
        )
        short_condition = (signals_df['short_signal_raw'] > threshold_short)

        signals_df['long_signal'] = long_condition.astype(float)
        signals_df['short_signal'] = short_condition.astype(float)

        self.stdout.write(self.style.WARNING("\n--- Final Signal Statistics ---"))
        actionable_longs = signals_df['long_signal'].sum()
        actionable_shorts = signals_df['short_signal'].sum()
        self.stdout.write(f"Actionable LONG signals (after filtering): {int(actionable_longs)}")
        self.stdout.write(f"Actionable SHORT signals (after filtering): {int(actionable_shorts)}")
        self.stdout.write(self.style.WARNING("---------------------------\n"))
        
        if actionable_longs == 0 and actionable_shorts == 0:
            self.stderr.write(self.style.ERROR("No trades would be executed after applying filters."))
            return

        self.stdout.write("Running backtest with filtered signals...")
        results = run_vectorized_backtest(prices_test, signals_df[['long_signal', 'short_signal']], config)
        
        html_path, json_path = generate_report(results, output_path_prefix=output_prefix)

        self.stdout.write(self.style.SUCCESS(f"\nBacktest complete. Reports saved to '{html_path}' and '{json_path}'"))
        self.stdout.write(f"Metrics: Total Return: {results.get('total_return_pct', 0):.2f}%, Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}, Trades: {results.get('num_trades', 0)}")

        