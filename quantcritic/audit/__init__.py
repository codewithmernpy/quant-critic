"""Robustness and integrity audit battery."""

from quantcritic.audit.costs import run_cost_stress_audit
from quantcritic.audit.lookahead import run_delay_test, run_lookahead_audit, run_truncation_test
from quantcritic.audit.montecarlo import (
    run_montecarlo_audit,
    run_random_entry_baseline,
    run_trade_bootstrap,
)
from quantcritic.audit.overfit import (
    compute_cscv_pbo,
    compute_deflated_sharpe_ratio,
    compute_parameter_stability,
    run_overfit_audit,
)
from quantcritic.audit.propfirm import run_propfirm_audit
from quantcritic.audit.regime import classify_btc_regimes, run_regime_audit
from quantcritic.audit.scorecard import (
    DEFAULT_STRICTNESS,
    STRICTNESS_MODES,
    STRICTNESS_RULES,
    compute_scorecard,
    normalize_strictness,
)
from quantcritic.audit.walkforward import (
    DataSplits,
    LockedSlice,
    run_walkforward_audit,
    run_window_backtest,
    split_market_data,
)

__all__ = [
    "DEFAULT_STRICTNESS",
    "DataSplits",
    "LockedSlice",
    "STRICTNESS_MODES",
    "STRICTNESS_RULES",
    "classify_btc_regimes",
    "compute_cscv_pbo",
    "compute_deflated_sharpe_ratio",
    "compute_parameter_stability",
    "compute_scorecard",
    "normalize_strictness",
    "run_cost_stress_audit",
    "run_delay_test",
    "run_lookahead_audit",
    "run_montecarlo_audit",
    "run_overfit_audit",
    "run_propfirm_audit",
    "run_random_entry_baseline",
    "run_regime_audit",
    "run_trade_bootstrap",
    "run_truncation_test",
    "run_walkforward_audit",
    "run_window_backtest",
    "split_market_data",
]
