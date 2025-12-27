# apps/analytics/backtesting/run.py
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def run_vectorized_backtest(
    prices_df: pd.DataFrame, 
    signals_df: pd.DataFrame, # الآن نستقبل DataFrame يحتوي على 'long_signal' و 'short_signal'
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Runs an event-driven backtest for pre-filtered long and short signals.
    """
    bt_cfg = config.get('backtesting', {})
    initial_capital = float(bt_cfg.get('initial_capital', 10000.0))
    sl_pct = bt_cfg.get('stop_loss_pct')
    tp_pct = bt_cfg.get('take_profit_pct')
    slippage = bt_cfg.get('slippage_pct', 0.0002)
    commission = bt_cfg.get('commission_pct', 0.0001)

    # **الإصلاح الحاسم هنا:** لم نعد بحاجة لحساب الإشارات، فهي جاهزة.
    # نقوم فقط بدمجها مع بيانات الأسعار.
    df = prices_df.copy().join(signals_df, how='inner').sort_index()
    
    # نستخدم shift(1) للنظر في إشارة الشمعة السابقة للدخول في شمعة الافتتاح الحالية
    df['enter_long'] = (df['long_signal'].shift(1) > 0)
    df['enter_short'] = (df['short_signal'].shift(1) > 0)
    
    # منع الدخول في صفقات متعارضة (يبقى كما هو)
    conflicting_signals = df['enter_long'] & df['enter_short']
    df.loc[conflicting_signals, ['enter_long', 'enter_short']] = False
    
    trades: List[Dict] = []
    equity = [initial_capital]
    dates = [df.index[0] if not df.empty else pd.Timestamp.now(tz='UTC')]
    
    in_position = 0
    position_details = {}

    for i in range(1, len(df)):
        current_row = df.iloc[i]
        
        # 1. التحقق من شروط الخروج (لا تغيير هنا)
        if in_position != 0:
            exit_price, exit_reason = None, None
            if in_position == 1:
                if sl_pct and current_row['low'] <= position_details.get('stop_price', -np.inf):
                    exit_price, exit_reason = position_details['stop_price'], 'SL'
                elif tp_pct and current_row['high'] >= position_details.get('tp_price', np.inf):
                    exit_price, exit_reason = position_details['tp_price'], 'TP'
            elif in_position == -1:
                if sl_pct and current_row['high'] >= position_details.get('stop_price', np.inf):
                    exit_price, exit_reason = position_details['stop_price'], 'SL'
                elif tp_pct and current_row['low'] <= position_details.get('tp_price', -np.inf):
                    exit_price, exit_reason = position_details['tp_price'], 'TP'

            if exit_price:
                entry_price_after_costs = position_details['entry_price_after_costs']
                if in_position == 1:
                    exit_price_after_costs = exit_price * (1 - slippage - commission)
                    pnl_pct = (exit_price_after_costs / entry_price_after_costs) - 1
                else:
                    exit_price_after_costs = exit_price * (1 + slippage + commission)
                    pnl_pct = (entry_price_after_costs / exit_price_after_costs) - 1

                trades.append({
                    'entry_time': position_details['entry_time'].isoformat(),
                    'exit_time': current_row.name.isoformat(), 'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason, 'trade_type': 'LONG' if in_position == 1 else 'SHORT'
                })
                equity.append(equity[-1] * (1 + pnl_pct))
                dates.append(current_row.name)
                in_position, position_details = 0, {}

        # 2. التحقق من إشارات الدخول الجديدة (لا تغيير هنا)
        if in_position == 0:
            entry_price = current_row['open']
            trade_type = None

            if current_row['enter_long']:
                trade_type = 1
                entry_price_after_costs = entry_price * (1 + slippage + commission)
                stop_price = entry_price * (1 - sl_pct) if sl_pct else None
                tp_price = entry_price * (1 + tp_pct) if tp_pct else None
            elif current_row['enter_short']:
                trade_type = -1
                entry_price_after_costs = entry_price * (1 - slippage - commission)
                stop_price = entry_price * (1 + sl_pct) if sl_pct else None
                tp_price = entry_price * (1 - tp_pct) if tp_pct else None

            if trade_type:
                in_position = trade_type
                position_details = {
                    'entry_time': current_row.name, 'entry_price_after_costs': entry_price_after_costs,
                    'stop_price': stop_price, 'tp_price': tp_price
                }

    # --- حساب المقاييس النهائية (لا تغيير هنا) ---
    equity_series = pd.Series(data=equity, index=pd.to_datetime(dates))
    if not trades:
        return {"total_return_pct": 0.0, "sharpe_ratio": 0.0, "max_drawdown_pct": 0.0, "final_capital": initial_capital, "equity_curve": {equity_series.index[0].isoformat(): initial_capital}, "num_trades": 0, "trades": []}

    final_capital = equity_series.iloc[-1]
    total_return = (final_capital / initial_capital) - 1
    trade_returns = pd.Series([t['pnl_pct'] for t in trades])
    
    annualization_factor = np.sqrt(252 * 24)
    sharpe_ratio = (trade_returns.mean() / (trade_returns.std() if trade_returns.std() != 0 else 1e-9)) * annualization_factor

    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown = drawdown.min()

    return {"total_return_pct": total_return * 100, "sharpe_ratio": sharpe_ratio, "max_drawdown_pct": max_drawdown * 100, "final_capital": final_capital, "equity_curve": {k.isoformat(): v for k, v in equity_series.to_dict().items()}, "num_trades": len(trades), "trades": trades}


def generate_report(results: Dict, output_path_prefix: str) -> Tuple[str, str]:
    # ... (لا تغيير هنا)
    equity_curve_data = {pd.to_datetime(k): v for k, v in results['equity_curve'].items()}
    equity_curve = pd.Series(equity_curve_data).sort_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode='lines', name='Equity'))
    fig.update_layout(title='Backtest Equity Curve', xaxis_title='Date', yaxis_title='Capital', template='plotly_dark')
    html_path = f"{output_path_prefix}.html"
    json_path = f"{output_path_prefix}.json"
    fig.write_html(html_path)
    report_data = {k: v for k, v in results.items() if k != 'equity_curve'}
    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    return html_path, json_path

    