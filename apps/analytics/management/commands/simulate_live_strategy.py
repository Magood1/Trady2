# apps/analytics/management/commands/simulate_live_strategy.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from django.core.management.base import BaseCommand
from django.conf import settings
from apps.analytics.models.train import load_training_data
from apps.market_data.models import Asset
from apps.trading_core.strategies import bollinger_breakout_signal
# قمنا باستيراد get_model_by_version
from apps.mlops.services import get_active_model, get_model_by_version
from datetime import datetime, timezone

class Command(BaseCommand):
    help = "Time Travel Simulation: Q4 2024 Realistic PnL Test"

    def add_arguments(self, parser):
        parser.add_argument('--symbol', type=str, default='USDJPY')
        parser.add_argument('--timeframe', type=str, default='H1')
        parser.add_argument('--initial_capital', type=float, default=10000.0)
        # الإضافة الجديدة: خيار لتحديد نسخة النموذج
        parser.add_argument('--model_version', type=str, required=False, help='Specific model version string to use (e.g. 2.0.0+...)')

    def handle(self, *args, **options):
        symbol = options['symbol']
        tf = options['timeframe']
        capital = options['initial_capital']
        target_version = options['model_version']
        
        self.stdout.write(self.style.WARNING(f"--- STARTING REALISTIC SIMULATION FOR {symbol} [Q4 2024] ---"))

        # 1. Load Model & Data
        model_info = None
        
        if target_version:
            self.stdout.write(f"Attempting to load specific model: {target_version}")
            model_info = get_model_by_version(target_version)
            if not model_info:
                self.stdout.write(self.style.ERROR(f"Model version '{target_version}' not found in registry."))
                return
        else:
            self.stdout.write("No version specified. Loading latest active model...")
            model_info = get_active_model()

        if not model_info:
            self.stdout.write(self.style.ERROR("No valid model found to run simulation."))
            return

        model, registry = model_info
        self.stdout.write(self.style.SUCCESS(f"Loaded Model: {registry.version}"))
        
        # --- (باقي الكود كما هو تماماً دون تغيير) ---
        try:
            asset = Asset.objects.get(symbol=symbol)
            X, prices_df = load_training_data(asset, tf)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Data Load Error: {e}"))
            return
        
        # 2. Filter for Out-of-Sample Period (Q4 2024)
        start_date = datetime(2024, 10, 1, tzinfo=timezone.utc)
        mask = X.index >= start_date
        
        if mask.sum() == 0:
             self.stdout.write(self.style.ERROR("No data found for Q4 2024."))
             return

        X_test = X[mask].copy()
        prices_test = prices_df[mask].copy()
        
        self.stdout.write(f"Simulating over {len(prices_test)} candles...")

        # 3. Simulation Loop
        equity = [capital]
        trades = []
        
        # Realistic Costs Configuration
        SPREAD_PCT = 0.00015
        COMMISSION_PCT = 0.00007
        
        # 3.1 Get Predictions
        X_input = pd.DataFrame()
        for col in registry.feature_list:
            X_input[col] = X_test.get(col, 0.0)
            
        preds_proba = model.predict_proba(X_input)[:, 1]
        
        # 3.2 Get Base Signals
        from ta.volatility import BollingerBands
        from ta.momentum import RSIIndicator
        
        bb = BollingerBands(close=prices_test['close'], window=20, window_dev=2)
        rsi_ind = RSIIndicator(close=prices_test['close'], window=14)
        
        rsi = rsi_ind.rsi()
        bb_upper = bb.bollinger_hband()
        
        base_signals = (prices_test['close'] > bb_upper) & (rsi < 75)
        
        # 3.3 Combine
        ml_threshold = 0.60
        final_signals = base_signals & (preds_proba > ml_threshold)
        
        signal_indices = np.where(final_signals)[0]
        self.stdout.write(f"Candidates: {base_signals.sum()} | Executed: {len(signal_indices)} (Filter Rate: {len(signal_indices)/base_signals.sum():.1%})")
        
        # 3.4 Execution Loop
        for idx in signal_indices:
            if idx + 13 >= len(prices_test): continue
            
            entry_idx = idx
            entry_time = prices_test.index[entry_idx]
            entry_price = prices_test['close'].iloc[entry_idx]
            volatility = X_test['vol_std'].iloc[entry_idx]
            
            stop_dist = entry_price * volatility * 2.0
            
            if stop_dist == 0: continue

            sl_price = entry_price - stop_dist
            tp_price = entry_price + (stop_dist * 1.5)
            
            future_window = prices_test.iloc[entry_idx+1 : entry_idx+13]
            
            exit_price = prices_test['close'].iloc[entry_idx+12]
            reason = "TIME"
            
            for i in range(len(future_window)):
                high = future_window['high'].iloc[i]
                low = future_window['low'].iloc[i]
                
                if low <= sl_price:
                    exit_price = sl_price
                    reason = "SL"
                    slippage = volatility * entry_price * 0.2 
                    exit_price -= slippage
                    break
                elif high >= tp_price:
                    exit_price = tp_price
                    reason = "TP"
                    break
            
            raw_ret = (exit_price - entry_price) / entry_price
            net_ret = raw_ret - SPREAD_PCT - COMMISSION_PCT
            
            current_equity = equity[-1]
            risk_amt = current_equity * 0.01
            
            stop_price_delta = entry_price - sl_price
            if stop_price_delta <= 0: continue
            
            pos_units = risk_amt / stop_price_delta
            pnl_amt = pos_units * (exit_price - entry_price)
            cost_amt = pos_units * entry_price * (SPREAD_PCT + COMMISSION_PCT)
            
            final_pnl = pnl_amt - cost_amt
            equity.append(current_equity + final_pnl)
            
            trades.append({
                'entry_time': entry_time,
                'reason': reason,
                'net_ret_pct': net_ret * 100,
                'pnl_usd': final_pnl,
                'equity': equity[-1]
            })

        # 4. Final Report
        if not trades:
            self.stdout.write(self.style.ERROR("No trades executed."))
            return

        df_trades = pd.DataFrame(trades)
        total_ret_pct = ((equity[-1] - capital) / capital) * 100
        
        equity_curve = np.array(equity)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_dd = drawdown.min() * 100
        
        win_rate = (df_trades['pnl_usd'] > 0).mean() * 100
        profit_factor = abs(df_trades[df_trades['pnl_usd'] > 0]['pnl_usd'].sum() / df_trades[df_trades['pnl_usd'] <= 0]['pnl_usd'].sum()) if len(df_trades[df_trades['pnl_usd'] <= 0]) > 0 else float('inf')

        self.stdout.write(self.style.SUCCESS("\n--- SIMULATION RESULTS (Q4 2024) ---"))
        self.stdout.write(f"Symbol:          {symbol}")
        self.stdout.write(f"Timeframe:       {tf}")
        self.stdout.write(f"Initial Capital: ${capital:,.2f}")
        self.stdout.write(f"Final Equity:    ${equity[-1]:,.2f}")
        
        ret_color = self.style.SUCCESS if total_ret_pct > 0 else self.style.ERROR
        self.stdout.write(ret_color(f"Total Return:    {total_ret_pct:.2f}%"))
        
        dd_color = self.style.WARNING if max_dd < -5 else self.style.SUCCESS
        self.stdout.write(dd_color(f"Max Drawdown:    {max_dd:.2f}%"))
        
        self.stdout.write(f"Total Trades:    {len(trades)}")
        self.stdout.write(f"Win Rate:        {win_rate:.2f}%")
        self.stdout.write(f"Profit Factor:   {profit_factor:.2f}")
        self.stdout.write("-" * 30)