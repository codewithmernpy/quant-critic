"""AuditSession state, strict test-set lock enforcement, and @beta_tool definitions."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from anthropic import beta_tool
from pydantic import BaseModel, Field

from quantcritic.audit.costs import run_cost_stress_audit
from quantcritic.audit.lookahead import run_lookahead_audit
from quantcritic.audit.montecarlo import run_montecarlo_audit
from quantcritic.audit.overfit import run_overfit_audit
from quantcritic.audit.propfirm import run_propfirm_audit
from quantcritic.audit.regime import run_regime_audit
from quantcritic.audit.scorecard import (
    cap_verdict,
    compute_scorecard,
    normalize_strictness,
    verdict_from_score,
)
from quantcritic.audit.walkforward import (
    DataSplits,
    run_walkforward_audit,
    run_window_backtest,
    split_market_data,
)
from quantcritic.config import DEFAULT_STRICTNESS, AuditConfig, BacktestConfig, PropFirmConfig, SplitConfig
from quantcritic.data.loader import MarketData, concat_market_data
from quantcritic.engine.backtest import BacktestResult, run_backtest
from quantcritic.strategies.base import Strategy

logger = logging.getLogger(__name__)

MAX_TOOL_RESPONSE_CHARS = 2900


class CriticVerdict(BaseModel):
    """Structured final critique produced by either Claude AI or the offline critic."""

    verdict: Literal["PROMISING", "NEEDS WORK", "REJECT"]
    score: float = Field(ge=0.0, le=100.0)
    summary: str
    strengths: list[str]
    weaknesses: list[str]
    red_flags: list[str]
    improvements: list[str]
    recommended_risk_per_trade: float = Field(ge=0.0, le=0.10)
    source: Literal["ai", "offline"] = "offline"


def to_jsonable(obj: Any, float_digits: int | None = None) -> Any:
    """Recursively convert numpy/pandas scalars to Python types and NaN/inf to None."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, float_digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, float_digits) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        if not np.isfinite(val):
            return None
        return round(val, float_digits) if float_digits is not None else val
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def _trim_lists(obj: Any, max_items: int) -> Any:
    if isinstance(obj, dict):
        return {k: _trim_lists(v, max_items) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_trim_lists(v, max_items) for v in obj[:max_items]]
    return obj


def _dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False)


def _clip_json(payload: dict[str, Any], limit: int = MAX_TOOL_RESPONSE_CHARS) -> str:
    """Serialize `payload` to compact, always-valid JSON of at most `limit` characters.

    Shrinks progressively (round floats, trim long lists, then drop the largest top-level
    entries) instead of cutting the string, so the model never receives broken JSON.
    """
    clean = to_jsonable(payload)
    text = _dumps(clean)
    if len(text) <= limit:
        return text
    for digits, max_items in ((4, 8), (4, 4), (3, 2)):
        cand = _trim_lists(to_jsonable(payload, float_digits=digits), max_items)
        text = _dumps(cand)
        if len(text) <= limit:
            return text
    cand = dict(cand) if isinstance(cand, dict) else {"value": cand}
    omitted: list[str] = []
    while len(_dumps({**cand, "_omitted": omitted})) > limit and cand:
        largest = max(cand, key=lambda k: len(_dumps(cand[k])))
        omitted.append(largest)
        cand.pop(largest)
    return _dumps({**cand, "_omitted": omitted}) if omitted else _dumps(cand)


@dataclass
class AuditSession:
    """Shared state and locked-test boundary for a single strategy audit."""

    strategy: Strategy
    data: MarketData
    config: BacktestConfig
    audit_config: AuditConfig = field(default_factory=AuditConfig)
    split_config: SplitConfig = field(default_factory=SplitConfig)
    prop_config: PropFirmConfig = field(default_factory=PropFirmConfig)
    warmup_data: MarketData | None = None
    strictness: Literal["strict", "balanced", "relaxed"] | None = None
    splits: DataSplits = field(init=False)
    chosen_params: dict[str, Any] = field(init=False)
    final_test_called: bool = False
    decisions_frozen: bool = False
    experiment_count: int = 0
    max_experiments: int = 5
    cached_results: dict[str, Any] = field(default_factory=dict)
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    agent_thoughts: list[str] = field(default_factory=list)
    verdict: CriticVerdict | None = None

    # Cached audits that depend on the currently chosen parameters
    _PARAM_DEPENDENT_CACHES = (
        "train_val_chosen",
        "lookahead",
        "overfit",
        "montecarlo",
        "costs",
        "regime",
        "propfirm",
        "scorecard",
    )

    def __post_init__(self) -> None:
        self.splits = split_market_data(self.data, self.split_config)
        self.chosen_params = dict(self.strategy.default_params)
        if self.strictness is None:
            self.strictness = normalize_strictness(self.audit_config.strictness)
        else:
            mode = normalize_strictness(self.strictness)
            self.strictness = mode
            if self.audit_config.strictness != mode:
                self.audit_config = self.audit_config.model_copy(update={"strictness": mode})

    def _with_warmup(self, md: MarketData) -> MarketData:
        if self.warmup_data is None:
            return md
        return concat_market_data(self.warmup_data, md)

    def _trade_start(self) -> pd.Timestamp | None:
        if self.warmup_data is None:
            return None
        return self.splits.train_start

    @property
    def test_access_count(self) -> int:
        """Number of times the locked TEST slice has actually been read (0 before final_test)."""
        return self.splits.test.access_count

    def _set_chosen_params(self, params: dict[str, Any]) -> None:
        """Change the working parameters and drop every cached audit computed with the old ones."""
        new = dict(params)
        if new != self.chosen_params:
            self.chosen_params = new
            for key in self._PARAM_DEPENDENT_CACHES:
                self.cached_results.pop(key, None)

    def _record_tool(self, tool_name: str, args: dict[str, Any], result_summary: str, elapsed_s: float) -> None:
        self.tool_log.append(
            {
                "step": len(self.tool_log) + 1,
                "tool": tool_name,
                "args": args,
                "summary": result_summary[:400],
                "elapsed_s": round(elapsed_s, 3),
            }
        )

    def get_overview(self) -> dict[str, Any]:
        """Return strategy metadata, parameter space, split boundaries, and train+val baseline."""
        t0 = time.perf_counter()
        if "overview" not in self.cached_results:
            tv_res = run_backtest(
                self._with_warmup(self.splits.train_val),
                self.strategy,
                self.chosen_params,
                self.config,
                trade_start=self._trade_start(),
            )
            self.cached_results["train_val_baseline"] = tv_res
            m = tv_res.metrics
            self.cached_results["overview"] = {
                "strategy": self.strategy.name,
                "description": self.strategy.description,
                "default_params": self.strategy.default_params,
                "param_grid": self.strategy.param_grid,
                "n_grid_combos": len(self.strategy.expand_param_grid()),
                "symbols": self.data.symbols,
                "interval": self.data.interval,
                "splits": {
                    "train_start": str(self.splits.train_start),
                    "train_end": str(self.splits.train_end),
                    "val_start": str(self.splits.val_start),
                    "val_end": str(self.splits.val_end),
                    "test_start": str(self.splits.test_start),
                    "test_end": str(self.splits.test_end),
                    "test_locked": True,
                },
                "train_val_headline_metrics": {
                    "total_return": m["total_return"],
                    "cagr": m["cagr"],
                    "sharpe": m["sharpe"],
                    "sortino": m["sortino"],
                    "max_drawdown": m["max_drawdown"],
                    "n_trades": m["n_trades"],
                    "win_rate": m["win_rate"],
                    "expectancy_r": m["expectancy_r"],
                    "profit_factor": m["profit_factor"],
                    "total_fees": m["total_fees"],
                    "total_funding": m["total_funding"],
                },
            }
        ov = self.cached_results["overview"]
        hm = ov["train_val_headline_metrics"]
        self._record_tool(
            "get_overview",
            {},
            f"Train+Val Sharpe={hm['sharpe']:.2f}, Ret={hm['total_return']*100:+.1f}%, Trades={hm['n_trades']}",
            time.perf_counter() - t0,
        )
        return ov

    def run_backtest_period(self, params_json: str, period: str = "train") -> dict[str, Any]:
        """Run a hypothesis-driven backtest strictly on train / validation / train+validation."""
        t0 = time.perf_counter()
        if self.decisions_frozen:
            err = {"error": "Decisions are frozen after final_test(); run_backtest is permanently locked."}
            self._record_tool("run_backtest", {"params_json": params_json, "period": period}, err["error"], 0.0)
            return err

        if period not in ("train", "validation", "train+validation"):
            err = {
                "error": (
                    f"Invalid period {period!r}. Allowed: 'train', 'validation', 'train+validation'. "
                    "The test slice is locked."
                )
            }
            self._record_tool("run_backtest", {"params_json": params_json, "period": period}, err["error"], 0.0)
            return err

        if self.experiment_count >= self.max_experiments:
            err = {"error": f"Experiment budget exhausted ({self.max_experiments} max hypothesis tests)."}
            self._record_tool("run_backtest", {"params_json": params_json, "period": period}, err["error"], 0.0)
            return err

        try:
            parsed = json.loads(params_json) if params_json else {}
            if not isinstance(parsed, dict):
                raise ValueError("params_json must decode to a JSON object")
        except Exception as exc:
            err = {"error": f"Invalid params_json: {exc}"}
            self._record_tool("run_backtest", {"params_json": params_json, "period": period}, err["error"], 0.0)
            return err

        self.experiment_count += 1
        merged = self.strategy.merge_params(parsed)
        if period == "validation":
            # Warm signals from train (plus any pre-range) history; only trades entered inside validation count
            res = run_window_backtest(
                self._with_warmup(self.splits.train_val),
                self.strategy,
                merged,
                self.config,
                self.splits.val_start,
            )
        else:
            data_map = {"train": self.splits.train, "train+validation": self.splits.train_val}
            res = run_backtest(
                self._with_warmup(data_map[period]),
                self.strategy,
                merged,
                self.config,
                trade_start=self._trade_start(),
            )
        m = res.metrics
        compact = {
            "experiment_number": self.experiment_count,
            "period": period,
            "params": merged,
            "metrics": {
                "total_return": m["total_return"],
                "cagr": m["cagr"],
                "sharpe": m["sharpe"],
                "sortino": m["sortino"],
                "max_drawdown": m["max_drawdown"],
                "n_trades": m["n_trades"],
                "win_rate": m["win_rate"],
                "expectancy_r": m["expectancy_r"],
                "profit_factor": m["profit_factor"],
                "total_fees": m["total_fees"],
                "total_funding": m["total_funding"],
            },
        }
        self._record_tool(
            "run_backtest",
            {"params": merged, "period": period},
            f"[{period}] Sharpe={m['sharpe']:.2f}, Ret={m['total_return']*100:+.1f}%, Trades={m['n_trades']}",
            time.perf_counter() - t0,
        )
        return compact

    def run_lookahead(self) -> dict[str, Any]:
        """Run truncation causality and 1-bar entry delay tests on train+validation."""
        t0 = time.perf_counter()
        if "lookahead" not in self.cached_results:
            # Delay-test baseline must use the same parameters as the delayed rerun
            base_tv = self._get_active_train_val_result()
            res = run_lookahead_audit(
                data=self._with_warmup(self.splits.train_val),
                strategy=self.strategy,
                params=self.chosen_params,
                config=self.config,
                n_cuts=self.audit_config.lookahead_cuts,
                window=self.audit_config.lookahead_window,
                seed=self.audit_config.seed,
                base_result=base_tv,
                trade_start=self._trade_start(),
            )
            self.cached_results["lookahead"] = res
        la = self.cached_results["lookahead"]
        self._record_tool("lookahead_test", {}, la["explanation"], time.perf_counter() - t0)
        return {
            "status": la["status"],
            "explanation": la["explanation"],
            "truncation": {
                "passed": la["truncation"]["passed"],
                "cuts_checked": la["truncation"]["cuts_checked"],
                "cuts_with_mismatch": la["truncation"]["cuts_with_mismatch"],
                "total_mismatches": la["truncation"]["total_mismatches"],
                "examples": la["truncation"]["examples"][:4],
            },
            "delay": la["delay"],
        }

    def run_walkforward(self) -> dict[str, Any]:
        """Run train grid optimization, train->val degradation, and rolling WF on train+val."""
        t0 = time.perf_counter()
        if "walkforward" not in self.cached_results:
            wf = run_walkforward_audit(
                splits=self.splits,
                strategy=self.strategy,
                config=self.config,
                audit_cfg=self.audit_config,
                warmup_data=self.warmup_data,
            )
            self.cached_results["walkforward"] = wf
            if not self.decisions_frozen and wf.get("best_train_params"):
                # Adopt the train-optimal parameters; audits cached with older params are dropped
                self._set_chosen_params(wf["best_train_params"])
        wf = self.cached_results["walkforward"]
        self._record_tool("walk_forward", {}, wf["explanation"], time.perf_counter() - t0)
        tr_m = wf["train_metrics"]
        va_m = wf["validation_metrics"]
        oos_m = wf["rolling_oos_metrics"]
        return {
            "status": wf["status"],
            "explanation": wf["explanation"],
            "best_train_params": wf["best_train_params"],
            "train": {
                "sharpe": tr_m["sharpe"],
                "total_return": tr_m["total_return"],
                "n_trades": tr_m["n_trades"],
                "expectancy_r": tr_m["expectancy_r"],
                "max_drawdown": tr_m["max_drawdown"],
            },
            "validation": {
                "sharpe": va_m["sharpe"],
                "total_return": va_m["total_return"],
                "n_trades": va_m["n_trades"],
                "expectancy_r": va_m["expectancy_r"],
                "max_drawdown": va_m["max_drawdown"],
            },
            "sharpe_degradation_pct": wf["sharpe_degradation_pct"],
            "rolling_wf_oos": {
                "n_folds": len(wf["rolling_folds"]),
                "sharpe": oos_m.get("sharpe", 0.0),
                "total_return": oos_m.get("total_return", 0.0),
                "max_drawdown": oos_m.get("max_drawdown", 0.0),
                "n_trades": oos_m.get("n_trades", 0),
            },
        }

    def _get_active_train_val_result(self) -> BacktestResult:
        if "train_val_chosen" in self.cached_results:
            return self.cached_results["train_val_chosen"]
        res = run_backtest(
            self._with_warmup(self.splits.train_val),
            self.strategy,
            self.chosen_params,
            self.config,
            trade_start=self._trade_start(),
        )
        self.cached_results["train_val_chosen"] = res
        return res

    def check_min_train_val_trades(self, min_trades: int = 5) -> int:
        """Verify train+validation has enough trades before touching the locked test slice."""
        if "walkforward" not in self.cached_results:
            self.run_walkforward()
        n_trades = int(self._get_active_train_val_result().metrics["n_trades"])
        if n_trades < min_trades:
            raise ValueError(
                f"Audit requires at least {min_trades} trades in the train+validation window "
                f"(found {n_trades}). Widen the date range or choose a more active interval/strategy."
            )
        return n_trades

    def run_overfit(self) -> dict[str, Any]:
        """Run DSR, CSCV PBO, and parameter stability on train+validation."""
        t0 = time.perf_counter()
        if "overfit" not in self.cached_results:
            wf_cache = self.cached_results.get("walkforward", {})
            wf_grid = wf_cache.get("grid_results")
            pre_sigs = wf_cache.get("_combo_signals")
            ov = run_overfit_audit(
                train_val_data=self._with_warmup(self.splits.train_val),
                strategy=self.strategy,
                best_params=self.chosen_params,
                config=self.config,
                n_blocks=self.audit_config.cscv_blocks,
                train_grid_results=wf_grid,
                precomputed_signals=pre_sigs,
                trade_start=self._trade_start(),
            )
            self.cached_results["overfit"] = ov
        ov = self.cached_results["overfit"]
        self._record_tool("overfit_tests", {}, ov["explanation"], time.perf_counter() - t0)
        return {
            "status": ov["status"],
            "explanation": ov["explanation"],
            "n_trials": ov["n_trials"],
            "dsr": ov["dsr"],
            "pbo": {k: v for k, v in ov["pbo"].items() if k != "logits"},
            "stability": ov["stability"],
        }

    def _ensure_montecarlo(self) -> dict[str, Any]:
        if "montecarlo" not in self.cached_results:
            tv_res = self._get_active_train_val_result()
            mc = run_montecarlo_audit(
                data=self.splits.train_val,
                real_result=tv_res,
                params=self.chosen_params,
                mc_simulations=self.audit_config.mc_simulations,
                random_runs=self.audit_config.random_baseline_runs,
                seed=self.audit_config.seed,
            )
            self.cached_results["montecarlo"] = mc
        return self.cached_results["montecarlo"]

    def run_monte_carlo(self) -> dict[str, Any]:
        """Return 1000x trade bootstrap distributions."""
        t0 = time.perf_counter()
        mc = self._ensure_montecarlo()
        bs = {k: v for k, v in mc["bootstrap"].items() if k not in ("return_samples", "max_dd_samples")}
        self._record_tool(
            "monte_carlo",
            {},
            f"Median ret={bs['median_return']*100:+.1f}%, p95 DD={bs['p95_max_drawdown']*100:.1f}%, P(loss)={bs['prob_loss']*100:.1f}%",
            time.perf_counter() - t0,
        )
        return {"status": mc["status"], "bootstrap": bs}

    def run_random_baseline(self) -> dict[str, Any]:
        """Return 200x matched random-entry permutation test vs real expectancy."""
        t0 = time.perf_counter()
        mc = self._ensure_montecarlo()
        rb = {k: v for k, v in mc["random_baseline"].items() if k != "baseline_expectancies"}
        self._record_tool(
            "random_baseline",
            {},
            f"Real exp={rb['real_expectancy_r']:+.3f}R vs random={rb['random_mean_expectancy_r']:+.3f}R (p={rb['p_value']:.3f})",
            time.perf_counter() - t0,
        )
        return {"status": mc["status"], "random_baseline": rb}

    def run_cost_stress(self) -> dict[str, Any]:
        """Run fee multiplier x slippage bps grid on train+validation."""
        t0 = time.perf_counter()
        if "costs" not in self.cached_results:
            cs = run_cost_stress_audit(
                data=self._with_warmup(self.splits.train_val),
                strategy=self.strategy,
                params=self.chosen_params,
                config=self.config,
                trade_start=self._trade_start(),
            )
            self.cached_results["costs"] = cs
        cs = self.cached_results["costs"]
        self._record_tool("cost_stress", {}, cs["explanation"], time.perf_counter() - t0)
        return {
            "status": cs["status"],
            "explanation": cs["explanation"],
            "base_expectancy_r": cs["base_expectancy_r"],
            "stress_2x_5bps_expectancy_r": cs["stress_2x_5bps_expectancy_r"],
            "break_even_cost_bps": cs["break_even_cost_bps"],
            "expectancy_matrix": cs["expectancy_matrix"],
            "fee_multipliers": cs["fee_multipliers"],
            "slippage_bps_grid": cs["slippage_bps_grid"],
        }

    def run_regime_breakdown(self) -> dict[str, Any]:
        """Run yearly, BTC regime, symbol share, long/short, and top-5% outlier check."""
        t0 = time.perf_counter()
        if "regime" not in self.cached_results:
            tv_res = self._get_active_train_val_result()
            reg = run_regime_audit(self._with_warmup(self.splits.train_val), tv_res)
            self.cached_results["regime"] = reg
        reg = self.cached_results["regime"]
        self._record_tool("regime_breakdown", {}, reg["explanation"], time.perf_counter() - t0)
        return reg

    def run_prop_firm(self) -> dict[str, Any]:
        """Run two-step prop-firm rolling challenge simulation on train+validation."""
        t0 = time.perf_counter()
        if "propfirm" not in self.cached_results:
            tv_res = self._get_active_train_val_result()
            pf = run_propfirm_audit(tv_res, self.prop_config)
            self.cached_results["propfirm"] = pf
        pf = self.cached_results["propfirm"]
        self._record_tool("prop_firm_check", {}, pf["explanation"], time.perf_counter() - t0)
        return {k: v for k, v in pf.items() if k != "config"}

    def run_final_test(self, params_json: str = "{}") -> dict[str, Any]:
        """Unlock and evaluate the held-out 20% TEST slice EXACTLY ONCE."""
        t0 = time.perf_counter()
        if self.final_test_called:
            err = {"error": "final_test() has already been executed. Locked TEST slice may only be evaluated ONCE."}
            self._record_tool("final_test", {"params_json": params_json}, err["error"], 0.0)
            return err

        try:
            parsed = json.loads(params_json) if params_json and params_json.strip() else {}
            if not isinstance(parsed, dict):
                raise ValueError("params_json must decode to a JSON object")
        except Exception as exc:
            err = {"error": f"Invalid params_json: {exc}"}
            self._record_tool("final_test", {"params_json": params_json}, err["error"], 0.0)
            return err

        final_params = self.strategy.merge_params(parsed if parsed else self.chosen_params)
        self._set_chosen_params(final_params)
        self.final_test_called = True
        self.decisions_frozen = True

        # The ONLY read of the locked slice (counted by LockedSlice.access_count)
        test_md = self.splits.test.unlock()
        test_start = min(df.index[0] for df in test_md.ohlcv.values() if not df.empty)
        # Signals are generated once on the full history ending at the test end (including any
        # pre-range warm-up history), so indicators and stateful strategies are warm when the test
        # window opens; only trades entered at bars >= test_start (or >= range_start) are evaluated.
        full_hist = self._with_warmup(self.data)
        signals = {sym: self.strategy.generate(df, final_params) for sym, df in full_hist.ohlcv.items()}
        test_res = run_window_backtest(
            full_hist, self.strategy, final_params, self.config, test_start, signals=signals
        )
        self.cached_results["final_test_result"] = test_res
        # Full continuous backtest over the active range with the frozen params (same signals)
        full_res = run_backtest(
            full_hist, signals, final_params, self.config, trade_start=self._trade_start()
        )
        self.cached_results["full_result"] = full_res

        m = test_res.metrics
        summary = {
            "status": "evaluated_once",
            "decisions_frozen": True,
            "final_params": final_params,
            "test_period": {
                "start": str(self.splits.test_start),
                "end": str(self.splits.test_end),
            },
            "test_metrics": {
                "total_return": m["total_return"],
                "cagr": m["cagr"],
                "sharpe": m["sharpe"],
                "sortino": m["sortino"],
                "max_drawdown": m["max_drawdown"],
                "n_trades": m["n_trades"],
                "win_rate": m["win_rate"],
                "expectancy_r": m["expectancy_r"],
                "profit_factor": m["profit_factor"],
                "total_fees": m["total_fees"],
                "total_funding": m["total_funding"],
            },
        }
        self.cached_results["final_test"] = summary
        self._record_tool(
            "final_test",
            {"params": final_params},
            f"[LOCKED TEST] Sharpe={m['sharpe']:.2f}, Ret={m['total_return']*100:+.1f}%, MaxDD={m['max_drawdown']*100:.1f}%, Trades={m['n_trades']}",
            time.perf_counter() - t0,
        )
        return summary

    def _apply_hard_gates(
        self, verdict: str, score: float, red_flags: list[str]
    ) -> tuple[str, float, list[str]]:
        """Cap a verdict/score with the integrity gate and (once computed) the scorecard gates."""
        v_norm = verdict
        score_f = float(max(0.0, min(100.0, score)))
        la = self.cached_results.get("lookahead")
        if la and la.get("status") == "fail":
            v_norm = "REJECT"
            score_f = min(score_f, 20.0)
            if not any("lookahead" in rf.lower() or "truncat" in rf.lower() for rf in red_flags):
                red_flags = [la["explanation"], *red_flags]
        sc = self.cached_results.get("scorecard")
        if sc:
            mode = str(sc.get("strictness") or self.strictness or DEFAULT_STRICTNESS)
            capped_v = cap_verdict(v_norm, sc["verdict_ceiling"])
            capped_s = min(score_f, min((float(g["cap"]) for g in sc["gates"]), default=100.0))
            capped_v = cap_verdict(capped_v, verdict_from_score(capped_s, strictness=mode))
            if capped_v != v_norm or capped_s < score_f:
                reason_text = " ".join(g["reason"] for g in sc["gates"]) if sc["gates"] else (
                    f"Score {capped_s:.1f} is below the {mode} threshold."
                )
                note = f"Hard gate applied ({mode} mode): {reason_text}"
                if note not in red_flags:
                    red_flags = [note, *red_flags]
            v_norm, score_f = capped_v, round(capped_s, 1)
        return v_norm, score_f, red_flags

    def enforce_hard_gates(self) -> None:
        """Re-apply hard gates to an already submitted verdict once the scorecard exists."""
        if self.verdict is None:
            return
        v, s, rf = self._apply_hard_gates(self.verdict.verdict, self.verdict.score, list(self.verdict.red_flags))
        if (v, s) != (self.verdict.verdict, self.verdict.score):
            self.verdict = self.verdict.model_copy(update={"verdict": v, "score": s, "red_flags": rf})
            self._record_tool(
                "hard_gate",
                {"verdict": v, "score": s},
                f"Submitted verdict capped to {v} ({s:.1f}/100) by scorecard hard gates",
                0.0,
            )

    def record_verdict(
        self,
        verdict: str,
        score: float,
        summary: str,
        strengths: list[str],
        weaknesses: list[str],
        red_flags: list[str],
        improvements: list[str],
        recommended_risk_per_trade: float,
        source: Literal["ai", "offline"] = "ai",
    ) -> dict[str, Any]:
        """Validate and store the submitted CriticVerdict."""
        t0 = time.perf_counter()
        v_norm = verdict.strip().upper()
        if v_norm not in ("PROMISING", "NEEDS WORK", "REJECT"):
            v_norm = "NEEDS WORK"

        v_norm, score_f, red_flags = self._apply_hard_gates(v_norm, score, list(red_flags))

        cv = CriticVerdict(
            verdict=v_norm,  # type: ignore[arg-type]
            score=score_f,
            summary=summary,
            strengths=list(strengths),
            weaknesses=list(weaknesses),
            red_flags=list(red_flags),
            improvements=list(improvements),
            recommended_risk_per_trade=float(max(0.0, min(0.10, recommended_risk_per_trade))),
            source=source,
        )
        self.verdict = cv
        self._record_tool(
            "submit_verdict",
            {"verdict": cv.verdict, "score": cv.score},
            f"Verdict={cv.verdict} ({cv.score:.1f}/100)",
            time.perf_counter() - t0,
        )
        return {"recorded": True, "verdict": cv.model_dump()}

    def ensure_complete_audit(self) -> dict[str, Any]:
        """Ensure all 8 audit modules + scorecard exist for the FINAL parameters.

        Walk-forward (parameter selection on train) runs first; every pre-test audit then runs on
        train+validation with the chosen parameters, before the locked test is opened. If the
        final test was run with different parameters (AI choice), the dropped audits are recomputed
        on train+validation for those final parameters so every reported number matches them.
        """
        steps: list[tuple[str, Callable[[], Any]]] = [
            ("walkforward", self.run_walkforward),
            ("overview", self.get_overview),
            ("lookahead", self.run_lookahead),
            ("overfit", self.run_overfit),
            ("montecarlo", self._ensure_montecarlo),
            ("costs", self.run_cost_stress),
            ("regime", self.run_regime_breakdown),
            ("propfirm", self.run_prop_firm),
        ]
        for key, fn in steps:
            if key not in self.cached_results:
                fn()
        if not self.final_test_called:
            self.run_final_test(json.dumps(self.chosen_params))
        for key, fn in steps:
            if key not in self.cached_results:
                fn()

        full_res: BacktestResult = self.cached_results["full_result"]
        scorecard = compute_scorecard(
            lookahead_res=self.cached_results["lookahead"],
            walkforward_res=self.cached_results["walkforward"],
            overfit_res=self.cached_results["overfit"],
            montecarlo_res=self.cached_results["montecarlo"],
            cost_res=self.cached_results["costs"],
            regime_res=self.cached_results["regime"],
            full_metrics=full_res.metrics,
            risk_per_trade=self.config.risk_per_trade,
            test_metrics=self.cached_results["final_test"]["test_metrics"],
            trades=full_res.trades,
            strictness=self.strictness or DEFAULT_STRICTNESS,
        )
        self.cached_results["scorecard"] = scorecard
        return scorecard


def build_beta_tools(
    session: AuditSession,
    on_tool_event: Callable[[str, dict[str, Any], str], None] | None = None,
) -> list[Any]:
    """Create the Anthropic @beta_tool list bound as closures over `session`."""

    def _emit(name: str, args: dict[str, Any], out_json: str) -> str:
        if on_tool_event is not None:
            on_tool_event(name, args, out_json)
        return out_json

    @beta_tool
    def get_overview() -> str:
        """Get strategy description, default parameters, data range, and train+val metrics."""
        out = _clip_json(session.get_overview())
        return _emit("get_overview", {}, out)

    @beta_tool
    def run_backtest(params_json: str, period: str = "train") -> str:
        """Run a backtest with custom params on train/validation/train+validation only.

        Args:
            params_json: JSON object string overriding default strategy parameters.
            period: One of 'train', 'validation', or 'train+validation'.
        """
        out = _clip_json(session.run_backtest_period(params_json=params_json, period=period))
        return _emit("run_backtest", {"params_json": params_json, "period": period}, out)

    @beta_tool
    def lookahead_test() -> str:
        """Run causal truncation (25 random cuts) and 1-bar entry delay integrity tests."""
        out = _clip_json(session.run_lookahead())
        return _emit("lookahead_test", {}, out)

    @beta_tool
    def walk_forward() -> str:
        """Optimize grid on train, measure validation decay, and run rolling 12m->3m WF."""
        out = _clip_json(session.run_walkforward())
        return _emit("walk_forward", {}, out)

    @beta_tool
    def overfit_tests() -> str:
        """Compute Deflated Sharpe Ratio, CSCV PBO (S=10), and parameter plateau stability."""
        out = _clip_json(session.run_overfit())
        return _emit("overfit_tests", {}, out)

    @beta_tool
    def monte_carlo() -> str:
        """Run 1000x trade-R bootstrap for return, 95th-pct max drawdown, and loss probability."""
        out = _clip_json(session.run_monte_carlo())
        return _emit("monte_carlo", {}, out)

    @beta_tool
    def random_baseline() -> str:
        """Run 200x matched random-entry portfolios and return p-value of real expectancy."""
        out = _clip_json(session.run_random_baseline())
        return _emit("random_baseline", {}, out)

    @beta_tool
    def cost_stress() -> str:
        """Run 4x4 fee multiplier x slippage bps grid and compute break-even cost per side."""
        out = _clip_json(session.run_cost_stress())
        return _emit("cost_stress", {}, out)

    @beta_tool
    def regime_breakdown() -> str:
        """Analyze performance by year, BTC regime (bull/bear/chop), symbol, and top-5% removal."""
        out = _clip_json(session.run_regime_breakdown())
        return _emit("regime_breakdown", {}, out)

    @beta_tool
    def prop_firm_check() -> str:
        """Simulate a generic 2-step prop-firm evaluation (+8%/+5%) across all start weeks."""
        out = _clip_json(session.run_prop_firm())
        return _emit("prop_firm_check", {}, out)

    @beta_tool
    def final_test(params_json: str = "{}") -> str:
        """Evaluate the locked 20% TEST slice ONCE with the finally chosen parameters.

        Args:
            params_json: JSON object of final strategy parameters; empty uses best train params.
        """
        out = _clip_json(session.run_final_test(params_json=params_json))
        return _emit("final_test", {"params_json": params_json}, out)

    @beta_tool
    def submit_verdict(
        verdict: str,
        score: float,
        summary: str,
        strengths: list[str],
        weaknesses: list[str],
        red_flags: list[str],
        improvements: list[str],
        recommended_risk_per_trade: float,
    ) -> str:
        """Submit the final structured audit verdict and evidence-backed critique.

        Args:
            verdict: Exactly one of 'PROMISING', 'NEEDS WORK', or 'REJECT'.
            score: Overall robustness score from 0 to 100.
            summary: Concise 2-4 sentence executive verdict citing concrete metrics.
            strengths: Bullet list of validated strengths with supporting numbers.
            weaknesses: Bullet list of weaknesses with supporting numbers.
            red_flags: Critical integrity, overfitting, cost, or crypto-specific risks.
            improvements: Actionable quantitative hypotheses for next iteration.
            recommended_risk_per_trade: Suggested fractional equity risk per trade (e.g. 0.005).
        """
        out = _clip_json(
            session.record_verdict(
                verdict=verdict,
                score=score,
                summary=summary,
                strengths=strengths,
                weaknesses=weaknesses,
                red_flags=red_flags,
                improvements=improvements,
                recommended_risk_per_trade=recommended_risk_per_trade,
                source="ai",
            )
        )
        return _emit("submit_verdict", {"verdict": verdict, "score": score}, out)

    return [
        get_overview,
        run_backtest,
        lookahead_test,
        walk_forward,
        overfit_tests,
        monte_carlo,
        random_baseline,
        cost_stress,
        regime_breakdown,
        prop_firm_check,
        final_test,
        submit_verdict,
    ]
