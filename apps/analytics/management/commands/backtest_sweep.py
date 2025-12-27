# apps/analytics/management/commands/backtest_sweep.py
import yaml
import pandas as pd
import numpy as np
import structlog
from django.core.management.base import BaseCommand, CommandParser
from django.utils import timezone
from apps.mlops.services import get_model_by_version
from apps.analytics.backtesting.run import run_vectorized_backtest
from apps.market_data.models import Asset, OHLCV
from apps.analytics.models.train import load_training_data

logger = structlog.get_logger(__name__)

class Command(BaseCommand):
    help = "Runs a backtest for a model across a sweep of thresholds and saves results."

    def add_arguments(self, parser: CommandParser):
        parser.add_argument("--model-version", type=str, required=True, help="Model version string to test.")
        parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
        parser.add_argument("--out-dir", type=str, default="backtest_sweep_results", help="Output directory for CSV/JSON results.")
        parser.add_argument("--min-th", type=float, default=0.20, help="Minimum threshold to sweep.")
        parser.add_argument("--max-th", type=float, default=0.35, help="Maximum threshold to sweep.")
        parser.add_argument("--step", type=float, default=0.01, help="Step size for threshold sweep.")

    def handle(self, *args, **options):
        now = timezone.now().strftime("%Y%m%d_%H%M%S")
        out_dir = options["out_dir"].rstrip("/")
        Path = None
        try:
            from pathlib import Path
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        model_version = options["model_version"]
        config_path = options["config"]

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.stdout.write(f"--- Starting Threshold Sweep for Model {model_version} ---")
        model_info = get_model_by_version(model_version)
        if not model_info:
            self.stderr.write(self.style.ERROR("Model version not found."))
            return
        model, registry = model_info

        train_cfg = config['training']
        asset = Asset.objects.get(symbol=train_cfg['asset_symbol'])
        all_ohlcv = OHLCV.objects.filter(asset=asset, timeframe=train_cfg['timeframe']).order_by('timestamp')
        if not all_ohlcv.exists():
            self.stderr.write(self.style.ERROR("No OHLCV data found for the asset/timeframe."))
            return

        prices_df = pd.DataFrame.from_records(all_ohlcv.values('timestamp', 'open', 'high', 'low', 'close'))
        prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], utc=True)
        prices_df = prices_df.set_index('timestamp')
        prices_df[['open', 'high', 'low', 'close']] = prices_df[['open', 'high', 'low', 'close']].apply(pd.to_numeric)

        X, _ = load_training_data(asset, train_cfg['timeframe'])

        predictions = model.predict_proba(X[registry.feature_list])[:, 1]
        signals = pd.Series(predictions, index=X.index)

        th_min = options["min_th"]
        th_max = options["max_th"]
        step = options["step"]
        threshold_sweep = np.arange(th_min, th_max + 1e-9, step)

        results_rows = []
        detailed_results = []

        for th in threshold_sweep:
            cfg_copy = dict(config)
            cfg_copy['backtesting'] = dict(config.get('backtesting', {}))
            cfg_copy['backtesting']['threshold'] = float(th)

            self.stdout.write(f"\nRunning backtest with threshold: {th:.3f} ...")
            results = run_vectorized_backtest(prices_df, signals, cfg_copy)

            num_trades = results.get('num_trades', 0)
            total_return = results.get('total_return_pct', 0.0)
            sharpe = results.get('sharpe_ratio', 0.0)
            max_dd = results.get('max_drawdown_pct', 0.0)

            row = {
                "threshold": th,
                "num_trades": int(num_trades),
                "total_return_pct": float(total_return),
                "sharpe_ratio": float(sharpe),
                "max_drawdown_pct": float(max_dd),
            }
            results_rows.append(row)

            # Save per-threshold detail (trades if available)
            trades = results.get('trades', [])
            detailed_results.append({"threshold": th, "trades": trades})

            if num_trades > 0:
                self.stdout.write(self.style.SUCCESS(f" => trades={num_trades}, return={total_return:.2f}%, sharpe={sharpe:.2f}, max_dd={max_dd:.2f}%"))
            else:
                self.stdout.write(self.style.NOTICE(" => no trades executed."))

        # Save summary CSV/JSON
        df_summary = pd.DataFrame(results_rows)
        csv_path = f"{out_dir}/threshold_sweep_summary_{now}.csv"
        json_path = f"{out_dir}/threshold_sweep_summary_{now}.json"
        df_summary.to_csv(csv_path, index=False)
        df_summary.to_json(json_path, orient="records", indent=2)

        # Save detailed trades
        detailed_path = f"{out_dir}/threshold_sweep_trades_{now}.json"
        import json
        with open(detailed_path, "w") as f:
            json.dump(detailed_results, f, indent=2, default=str)

        self.stdout.write(self.style.SUCCESS(f"\n--- Threshold Sweep Complete ---"))
        self.stdout.write(self.style.SUCCESS(f"Summary CSV: {csv_path}"))
        self.stdout.write(self.style.SUCCESS(f"Summary JSON: {json_path}"))
        self.stdout.write(self.style.SUCCESS(f"Trades JSON: {detailed_path}"))
        self.stdout.write(self.style.SUCCESS("Use the CSV to sort/filter thresholds (e.g., sharpe > 0, num_trades >= 10)."))

