"""Portfolio backtest engine and performance metrics."""

from quantcritic.engine.backtest import BacktestResult, PreparedMarket, prepare_market, run_backtest
from quantcritic.engine.metrics import compute_metrics, compute_monthly_returns_table

__all__ = [
    "BacktestResult",
    "PreparedMarket",
    "compute_metrics",
    "compute_monthly_returns_table",
    "prepare_market",
    "run_backtest",
]
