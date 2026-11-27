"""No-code rule-based strategy specification, causal indicator evaluator, and Claude NL parser."""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import anthropic
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from quantcritic.config import (
    DEFAULT_MODEL,
    USER_STRATEGIES_DIR,
    USER_STRATEGY_EXAMPLES_DIR,
)
from quantcritic.strategies.base import (
    Strategy,
    build_signal_frame,
    compute_atr,
    compute_ema,
    compute_rsi,
)

logger = logging.getLogger(__name__)

IndicatorName = Literal[
    "close",
    "open",
    "high",
    "low",
    "ema",
    "sma",
    "rsi",
    "atr",
    "donchian_upper",
    "donchian_lower",
    "bb_upper",
    "bb_lower",
    "bb_mid",
    "bb_bw_pct",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "volume_z",
    "funding_z",
    "roc",
]

OperatorName = Literal[
    "crosses_above",
    "crosses_below",
    "greater_than",
    "less_than",
]

CombineMode = Literal["all", "any"]

SUPPORTED_INDICATORS: tuple[str, ...] = (
    "close",
    "open",
    "high",
    "low",
    "ema",
    "sma",
    "rsi",
    "atr",
    "donchian_upper",
    "donchian_lower",
    "bb_upper",
    "bb_lower",
    "bb_mid",
    "bb_bw_pct",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "volume_z",
    "funding_z",
    "roc",
)

SUPPORTED_OPERATORS: tuple[str, ...] = (
    "crosses_above",
    "crosses_below",
    "greater_than",
    "less_than",
)

INDICATOR_LABELS: dict[str, str] = {
    "close": "Close Price",
    "open": "Open Price",
    "high": "High Price",
    "low": "Low Price",
    "ema": "EMA(n)",
    "sma": "SMA(n)",
    "rsi": "RSI(n)",
    "atr": "ATR(n)",
    "donchian_upper": "Donchian Upper(n) [prior n bars]",
    "donchian_lower": "Donchian Lower(n) [prior n bars]",
    "bb_upper": "Bollinger Upper(n, k)",
    "bb_lower": "Bollinger Lower(n, k)",
    "bb_mid": "Bollinger Mid(n)",
    "bb_bw_pct": "Bollinger Bandwidth Percentile(n, k, lookback) [0-100]",
    "macd_line": "MACD Line(fast, slow, signal)",
    "macd_signal": "MACD Signal(fast, slow, signal)",
    "macd_hist": "MACD Histogram(fast, slow, signal)",
    "volume_z": "Rolling Volume Z-Score(n)",
    "funding_z": "Trailing Funding-Rate Z-Score(n)",
    "roc": "Rate of Change ROC(n) [%]",
}

_SPEC_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
_NON_DIRECTIONAL_INDICATORS = {"bb_bw_pct", "atr", "volume_z"}
_SIGNED_ZERO_INDICATORS = {"macd_line", "macd_signal", "macd_hist", "funding_z", "roc"}
_OP_FLIP: dict[str, OperatorName] = {
    "crosses_above": "crosses_below",
    "crosses_below": "crosses_above",
    "greater_than": "less_than",
    "less_than": "greater_than",
}
_IND_MIRROR: dict[str, IndicatorName] = {
    "donchian_upper": "donchian_lower",
    "donchian_lower": "donchian_upper",
    "bb_upper": "bb_lower",
    "bb_lower": "bb_upper",
    "high": "low",
    "low": "high",
}


class IndicatorOperand(BaseModel):
    """Causal indicator operand evaluated strictly on bars <= t."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["indicator"] = "indicator"
    indicator: IndicatorName
    period: int = Field(default=14, ge=1, le=1000)
    std_dev: float = Field(default=2.0, gt=0.0, le=10.0)
    lookback: int = Field(default=120, ge=5, le=2000)
    fast_period: int = Field(default=12, ge=1, le=500)
    slow_period: int = Field(default=26, ge=2, le=1000)
    signal_period: int = Field(default=9, ge=1, le=500)

    @field_validator("std_dev")
    @classmethod
    def _validate_std_dev(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("std_dev must be finite")
        return float(v)

    @model_validator(mode="after")
    def _validate_indicator_params(self) -> "IndicatorOperand":
        if self.indicator in ("rsi", "bb_upper", "bb_lower", "bb_mid", "bb_bw_pct", "volume_z", "funding_z"):
            if self.period < 2:
                raise ValueError(f"{self.indicator} requires period >= 2 (got {self.period})")
        if self.indicator in ("macd_line", "macd_signal", "macd_hist"):
            if self.fast_period >= self.slow_period:
                raise ValueError(
                    f"MACD fast_period ({self.fast_period}) must be strictly less than slow_period ({self.slow_period})"
                )
        return self


class NumberOperand(BaseModel):
    """Constant numeric threshold operand."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["number"] = "number"
    value: float

    @field_validator("value")
    @classmethod
    def _validate_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Numeric threshold must be finite")
        return float(v)


RightOperand = Annotated[Union[IndicatorOperand, NumberOperand], Field(discriminator="kind")]


class Condition(BaseModel):
    """Single causal comparison rule evaluated on closed bars."""

    model_config = ConfigDict(extra="forbid")

    left: IndicatorOperand
    operator: OperatorName
    right: RightOperand


class ExitSpec(BaseModel):
    """Exit and risk rules for a no-code strategy."""

    model_config = ConfigDict(extra="forbid")

    atr_period: int = Field(default=14, ge=1, le=500)
    atr_stop_mult: float = Field(..., gt=0.0, le=50.0)
    tp_r: float | None = Field(default=None, gt=0.0, le=100.0)
    max_bars: int | None = Field(default=None, ge=1, le=10000)
    exit_rules: list[Condition] = Field(default_factory=list)
    exit_combine: CombineMode = "any"

    @field_validator("atr_stop_mult", "tp_r")
    @classmethod
    def _validate_finite_opt(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError("Exit float parameter must be finite")
        return float(v) if v is not None else None


class StrategySpec(BaseModel):
    """Serializable specification for a causal no-code QuantCritic strategy."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    long_rules: list[Condition] = Field(default_factory=list)
    short_rules: list[Condition] = Field(default_factory=list)
    mirror_long_for_short: bool = False
    combine: CombineMode = "all"
    exit: ExitSpec
    notes: str = ""
    unsupported: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean.startswith("custom:"):
            clean = clean[len("custom:") :].strip()
        if not _SPEC_NAME_RE.match(clean):
            raise ValueError(
                f"Invalid strategy name {v!r}: use 1-48 lowercase alphanumeric characters, '_' or '-'."
            )
        if clean == "leaky_demo":
            raise ValueError("Strategy name 'leaky_demo' is reserved.")
        return clean

    @model_validator(mode="after")
    def _validate_rules(self) -> "StrategySpec":
        if self.mirror_long_for_short:
            if not self.long_rules:
                raise ValueError("mirror_long_for_short=True requires at least one long rule.")
        elif not self.long_rules and not self.short_rules:
            raise ValueError("At least one long rule or short rule must be defined.")
        return self

    def effective_short_rules(self) -> list[Condition]:
        """Return explicit short rules or mirrored long rules when mirror_long_for_short is True."""
        if self.mirror_long_for_short:
            return [mirror_condition(c) for c in self.long_rules]
        return list(self.short_rules)


def _mirror_indicator_op(op: IndicatorOperand) -> IndicatorOperand:
    mirrored_name = _IND_MIRROR.get(op.indicator, op.indicator)
    if mirrored_name == op.indicator:
        return op
    return op.model_copy(update={"indicator": mirrored_name})


def mirror_condition(cond: Condition) -> Condition:
    """Symmetrically mirror a long condition into its corresponding short condition."""
    left_name = cond.left.indicator
    if isinstance(cond.right, NumberOperand):
        val = float(cond.right.value)
        if left_name in _NON_DIRECTIONAL_INDICATORS:
            return cond.model_copy(deep=True)
        if left_name == "rsi":
            mirrored_val = round(100.0 - val, 6)
            return Condition(
                left=cond.left,
                operator=_OP_FLIP[cond.operator],
                right=NumberOperand(value=mirrored_val),
            )
        if left_name in _SIGNED_ZERO_INDICATORS:
            mirrored_val = 0.0 if val == 0.0 else -val
            return Condition(
                left=cond.left,
                operator=_OP_FLIP[cond.operator],
                right=NumberOperand(value=mirrored_val),
            )
        return Condition(
            left=_mirror_indicator_op(cond.left),
            operator=_OP_FLIP[cond.operator],
            right=cond.right,
        )

    return Condition(
        left=_mirror_indicator_op(cond.left),
        operator=_OP_FLIP[cond.operator],
        right=_mirror_indicator_op(cond.right),
    )


def _operand_warmup(op: IndicatorOperand | NumberOperand) -> int:
    if isinstance(op, NumberOperand):
        return 0
    ind = op.indicator
    if ind in ("close", "open", "high", "low"):
        return 1
    if ind in ("ema", "sma", "volume_z", "funding_z"):
        return int(op.period)
    if ind in ("rsi", "atr", "donchian_upper", "donchian_lower", "roc"):
        return int(op.period) + 1
    if ind in ("bb_upper", "bb_lower", "bb_mid"):
        return int(op.period)
    if ind == "bb_bw_pct":
        return int(op.period) + int(op.lookback) - 1
    if ind in ("macd_line", "macd_signal", "macd_hist"):
        return int(op.slow_period) + int(op.signal_period)
    return 1


def _indicator_cache_key(op: IndicatorOperand) -> tuple[Any, ...]:
    ind = op.indicator
    if ind in ("close", "open", "high", "low"):
        return (ind,)
    if ind in ("ema", "sma", "rsi", "atr", "donchian_upper", "donchian_lower", "volume_z", "funding_z", "roc"):
        return (ind, int(op.period))
    if ind in ("bb_upper", "bb_lower", "bb_mid"):
        return (ind, int(op.period), float(op.std_dev) if ind != "bb_mid" else 0.0)
    if ind == "bb_bw_pct":
        return (ind, int(op.period), float(op.std_dev), int(op.lookback))
    if ind in ("macd_line", "macd_signal", "macd_hist"):
        return (ind, int(op.fast_period), int(op.slow_period), int(op.signal_period))
    return (ind,)


def compute_indicator_series(df: pd.DataFrame, op: IndicatorOperand) -> pd.Series:
    """Compute a strictly causal indicator Series on bars <= t."""
    ind = op.indicator
    close = df["close"].astype(float)

    if ind == "close":
        return close
    if ind == "open":
        return df["open"].astype(float)
    if ind == "high":
        return df["high"].astype(float)
    if ind == "low":
        return df["low"].astype(float)
    if ind == "ema":
        return compute_ema(close, int(op.period))
    if ind == "sma":
        n = int(op.period)
        return close.rolling(n, min_periods=n).mean()
    if ind == "rsi":
        return compute_rsi(close, int(op.period))
    if ind == "atr":
        return compute_atr(df, int(op.period))
    if ind == "donchian_upper":
        n = int(op.period)
        return df["high"].astype(float).shift(1).rolling(n, min_periods=n).max()
    if ind == "donchian_lower":
        n = int(op.period)
        return df["low"].astype(float).shift(1).rolling(n, min_periods=n).min()
    if ind in ("bb_upper", "bb_lower", "bb_mid", "bb_bw_pct"):
        n = int(op.period)
        k = float(op.std_dev)
        mid = close.rolling(n, min_periods=n).mean()
        if ind == "bb_mid":
            return mid
        sd = close.rolling(n, min_periods=n).std(ddof=0)
        if ind == "bb_upper":
            return mid + k * sd
        if ind == "bb_lower":
            return mid - k * sd
        bandwidth = (2.0 * k * sd) / mid.replace(0.0, np.nan)
        lb = int(op.lookback)
        return bandwidth.rolling(lb, min_periods=lb).rank(pct=True) * 100.0
    if ind in ("macd_line", "macd_signal", "macd_hist"):
        ema_f = compute_ema(close, int(op.fast_period))
        ema_s = compute_ema(close, int(op.slow_period))
        line = ema_f - ema_s
        if ind == "macd_line":
            return line
        sig = compute_ema(line, int(op.signal_period))
        if ind == "macd_signal":
            return sig
        return line - sig
    if ind in ("volume_z", "funding_z"):
        n = int(op.period)
        if ind == "volume_z":
            raw = df["volume"].astype(float)
        else:
            raw = (
                df["funding_rate"].astype(float).fillna(0.0)
                if "funding_rate" in df.columns
                else pd.Series(0.0, index=df.index, dtype=float)
            )
        mean = raw.rolling(n, min_periods=n).mean()
        std = raw.rolling(n, min_periods=n).std(ddof=1)
        z = (raw - mean) / std.where(std > 1e-12, np.nan)
        # Keep pre-warmup NaNs as NaN, while flat zero-variance regimes post-warmup evaluate to 0.0
        return z.where(mean.isna() | z.notna(), 0.0)
    if ind == "roc":
        n = int(op.period)
        prev = close.shift(n)
        return ((close / prev.where(prev != 0.0, np.nan)) - 1.0) * 100.0
    raise ValueError(f"Unsupported indicator: {ind!r}")


def _evaluate_condition(
    df: pd.DataFrame,
    cond: Condition,
    cache: dict[tuple[Any, ...], pd.Series],
) -> pd.Series:
    l_key = _indicator_cache_key(cond.left)
    if l_key not in cache:
        cache[l_key] = compute_indicator_series(df, cond.left)
    left_s = cache[l_key]

    if isinstance(cond.right, NumberOperand):
        right_s: pd.Series | float = float(cond.right.value)
        prev_right: pd.Series | float = float(cond.right.value)
    else:
        r_key = _indicator_cache_key(cond.right)
        if r_key not in cache:
            cache[r_key] = compute_indicator_series(df, cond.right)
        right_s = cache[r_key]
        prev_right = right_s.shift(1)

    op = cond.operator
    if op == "greater_than":
        res = left_s > right_s
    elif op == "less_than":
        res = left_s < right_s
    elif op == "crosses_above":
        res = (left_s > right_s) & (left_s.shift(1) <= prev_right)
    elif op == "crosses_below":
        res = (left_s < right_s) & (left_s.shift(1) >= prev_right)
    else:
        raise ValueError(f"Unsupported operator: {op!r}")
    return res.fillna(False).astype(bool)


def _combine_conditions(
    df: pd.DataFrame,
    rules: list[Condition],
    combine: CombineMode,
    cache: dict[tuple[Any, ...], pd.Series],
) -> pd.Series:
    if not rules:
        return pd.Series(False, index=df.index, dtype=bool)
    evaluated = [_evaluate_condition(df, c, cache) for c in rules]
    acc = evaluated[0]
    for nxt in evaluated[1:]:
        acc = (acc & nxt) if combine == "all" else (acc | nxt)
    return acc


def format_condition(cond: Condition) -> str:
    """Format a Condition into a concise human-readable string."""
    def _fmt_ind(op: IndicatorOperand) -> str:
        i = op.indicator
        if i in ("close", "open", "high", "low"):
            return i
        if i in ("bb_upper", "bb_lower"):
            return f"{i}({op.period}, {op.std_dev:g})"
        if i == "bb_bw_pct":
            return f"bb_bw_pct({op.period}, {op.std_dev:g}, lb={op.lookback})"
        if i in ("macd_line", "macd_signal", "macd_hist"):
            return f"{i}({op.fast_period},{op.slow_period},{op.signal_period})"
        return f"{i}({op.period})"

    r_str = f"{cond.right.value:g}" if isinstance(cond.right, NumberOperand) else _fmt_ind(cond.right)
    return f"{_fmt_ind(cond.left)} {cond.operator} {r_str}"


class BuilderStrategy(Strategy):
    """QuantCritic Strategy executing a validated causal StrategySpec."""

    def __init__(self, spec: StrategySpec) -> None:
        self.spec = spec
        self.name = f"custom:{spec.name}"
        self.description = spec.description or f"Custom no-code strategy ({spec.name})"
        atr_p = int(spec.exit.atr_period)
        atr_mult = float(spec.exit.atr_stop_mult)
        tp_r_val = float(spec.exit.tp_r) if spec.exit.tp_r is not None else 0.0
        max_bars_val = int(spec.exit.max_bars) if spec.exit.max_bars is not None else 0

        self.default_params = {
            "atr_period": atr_p,
            "atr_stop_mult": atr_mult,
            "tp_r": tp_r_val,
            "max_bars": max_bars_val,
        }
        mults = sorted({round(max(0.5, atr_mult * 0.75), 2), round(atr_mult, 2), round(atr_mult * 1.25, 2)})
        periods = sorted({max(5, atr_p - 4), atr_p, atr_p + 4})
        self.param_grid = {
            "atr_period": periods,
            "atr_stop_mult": mults,
        }

    def is_valid_params(self, params: dict[str, Any]) -> bool:
        return int(params.get("atr_period", 0)) >= 1 and float(params.get("atr_stop_mult", 0.0)) > 0.0

    def compute_warmup_bars(self, atr_period: int | None = None) -> int:
        """Compute deterministic causal warm-up bars strictly from spec periods."""
        eff_atr = int(atr_period if atr_period is not None else self.spec.exit.atr_period)
        short_rules = self.spec.effective_short_rules()
        all_conds = [*self.spec.long_rules, *short_rules, *self.spec.exit.exit_rules]
        req = eff_atr + 1
        for cond in all_conds:
            cross_extra = 1 if cond.operator in ("crosses_above", "crosses_below") else 0
            req = max(req, _operand_warmup(cond.left) + cross_extra)
            if isinstance(cond.right, IndicatorOperand):
                req = max(req, _operand_warmup(cond.right) + cross_extra)
        return req + 1

    def generate(self, df: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
        p = self.merge_params(params)
        atr_period = max(1, int(p["atr_period"]))
        atr_mult = float(p["atr_stop_mult"])
        tp_r = float(p.get("tp_r", 0.0) or 0.0)
        max_bars = max(0, int(p.get("max_bars", 0) or 0))

        n = len(df)
        if n == 0:
            return build_signal_frame(
                index=df.index,
                signal=np.zeros(0, dtype=np.int8),
                stop_dist=np.zeros(0, dtype=np.float64),
                tp_r=tp_r if tp_r > 0 else np.nan,
                max_bars=max_bars,
                exit_signal=np.zeros(0, dtype=bool),
            )

        cache: dict[tuple[Any, ...], pd.Series] = {}
        short_rules = self.spec.effective_short_rules()

        long_cond = _combine_conditions(df, self.spec.long_rules, self.spec.combine, cache)
        short_cond = _combine_conditions(df, short_rules, self.spec.combine, cache)
        exit_cond = _combine_conditions(df, self.spec.exit.exit_rules, self.spec.exit.exit_combine, cache)

        # Trigger events only (False->True transitions) so state conditions do not refire every bar
        long_entry = long_cond & (~long_cond.shift(1, fill_value=False))
        short_entry = short_cond & (~short_cond.shift(1, fill_value=False))
        conflict = long_entry & short_entry

        sig = np.zeros(n, dtype=np.int8)
        sig[(long_entry & ~conflict).to_numpy()] = 1
        sig[(short_entry & ~conflict).to_numpy()] = -1
        ex_sig = exit_cond.to_numpy(dtype=bool).copy()

        warmup = self.compute_warmup_bars(atr_period=atr_period)
        cut = min(warmup, n)
        sig[:cut] = 0
        ex_sig[:cut] = False

        atr = compute_atr(df, atr_period)
        stop_dist = (atr * atr_mult).to_numpy(dtype=np.float64)
        return build_signal_frame(
            index=df.index,
            signal=sig,
            stop_dist=stop_dist,
            tp_r=tp_r if tp_r > 0 else np.nan,
            max_bars=max_bars,
            exit_signal=ex_sig,
        )


def save_strategy_spec(spec: StrategySpec, directory: Path | None = None) -> Path:
    """Validate and save a StrategySpec JSON file into `user_strategies/<name>.json`."""
    validated = StrategySpec.model_validate(spec.model_dump())
    target_dir = directory or USER_STRATEGIES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"{validated.name}.json"
    out_path.write_text(validated.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return out_path


def load_strategy_spec(path: Path) -> StrategySpec:
    """Load and validate a StrategySpec from a JSON file."""
    return StrategySpec.model_validate_json(path.read_text(encoding="utf-8"))


def load_all_spec_strategies(
    user_dir: Path | None = None,
    examples_dir: Path | None = None,
) -> dict[str, BuilderStrategy]:
    """Load example and user JSON strategy specs into `custom:<name>` BuilderStrategy instances."""
    u_dir = user_dir or USER_STRATEGIES_DIR
    ex_dir = examples_dir or USER_STRATEGY_EXAMPLES_DIR
    loaded: dict[str, BuilderStrategy] = {}

    for directory in (ex_dir, u_dir):
        if not directory.is_dir():
            continue
        for json_path in sorted(directory.glob("*.json")):
            try:
                spec = load_strategy_spec(json_path)
                strat = BuilderStrategy(spec)
                loaded[strat.name] = strat
            except Exception as exc:
                logger.warning("Skipping invalid custom strategy spec %s: %s", json_path, exc)
    return loaded


SPEC_PROMPT = """You translate plain-English crypto perpetual trading ideas into a strictly causal QuantCritic `StrategySpec` JSON object.

RULES & CONSTRAINTS:
1. CLOSED-BAR CAUSAL LOGIC ONLY: Every condition is evaluated at bar close `t` using only bars <= t, and entries/exits execute at bar `t+1` open. Never express future bars, negative shifts, centered windows, or full-sample statistics.
2. SUPPORTED INDICATORS ONLY:
   - Price: `close`, `open`, `high`, `low`
   - Moving averages: `ema` (period), `sma` (period)
   - Oscillators / volatility: `rsi` (period >= 2), `atr` (period >= 1), `roc` (percentage rate of change over `period` bars)
   - Channels: `donchian_upper` / `donchian_lower` (channel of the PREVIOUS `period` bars, shifted by 1)
   - Bollinger Bands: `bb_upper`, `bb_lower`, `bb_mid` (period >= 2, std_dev > 0), `bb_bw_pct` (0-100 trailing percentile of Bollinger bandwidth over `lookback` bars; e.g. < 20 means tight squeeze)
   - MACD: `macd_line`, `macd_signal`, `macd_hist` (`fast_period` < `slow_period`, `signal_period`)
   - Z-scores (trailing): `volume_z` (period >= 2), `funding_z` (trailing funding-rate z-score over `period` >= 2 bars)
3. SUPPORTED OPERATORS ONLY:
   - `crosses_above`, `crosses_below`, `greater_than`, `less_than`
4. EXITS & ATR STOP REQUIRED:
   - Always provide an ATR protective stop via `exit.atr_period` and `exit.atr_stop_mult` (> 0). If the user omitted a stop, choose a reasonable ATR stop (e.g., atr_period=14, atr_stop_mult=2.0) and explain that choice in `notes`.
   - You may also set `exit.tp_r` (optional R-multiple take profit), `exit.max_bars` (optional time stop), and `exit.exit_rules` (optional causal close-at-next-open conditions).
5. NAME & METADATA:
   - Set `name` to a short lowercase snake_case slug (matching `^[a-z0-9][a-z0-9_-]{0,47}$`).
   - List EVERY requirement from the user description that cannot be expressed with the supported indicators, operators, or exit types in `unsupported` (e.g., trailing stops, order-book/liquidation feeds, multi-timeframe rules, limit orders, position pyramiding), and explain assumptions in `notes`.
   - If the user asks for short symmetry or describes only longs with symmetric phrasing, set `mirror_long_for_short=true` (or specify explicit `short_rules`)."""


@dataclass
class ParseSpecResult:
    """Outcome of translating plain English into a StrategySpec via Claude."""

    spec: StrategySpec | None = None
    error: str | None = None
    refusal: bool = False


def parse_strategy_description(
    description: str,
    api_key: str | None = None,
    model: str | None = None,
    client: Any | None = None,
) -> ParseSpecResult:
    """Translate a plain-English trading idea into a StrategySpec using Claude structured output."""
    text = description.strip()
    if not text:
        return ParseSpecResult(error="Please enter a strategy description first.")

    effective_key = (api_key or os.getenv("ANTHROPIC_API_KEY", "")).strip()
    if client is None and not effective_key:
        return ParseSpecResult(
            error="No Anthropic API key configured. Enter one in the sidebar or set ANTHROPIC_API_KEY."
        )

    chosen_model = (model or os.getenv("QC_MODEL", DEFAULT_MODEL)).strip() or DEFAULT_MODEL
    active_client = client if client is not None else anthropic.Anthropic(api_key=effective_key)

    try:
        response = active_client.beta.messages.parse(
            model=chosen_model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "high"},
            system=SPEC_PROMPT,
            messages=[{"role": "user", "content": text}],
            output_format=StrategySpec,
        )
    except ValidationError as exc:
        return ParseSpecResult(error=f"Generated spec failed validation: {exc}")
    except anthropic.APIError as exc:
        return ParseSpecResult(error=f"Anthropic API error ({type(exc).__name__}): {exc}")

    if getattr(response, "stop_reason", None) == "refusal":
        return ParseSpecResult(
            refusal=True,
            error="Claude declined to parse this prompt (stop_reason='refusal'). Adjust your description or build manually.",
        )

    parsed = getattr(response, "parsed_output", None)
    if parsed is None:
        return ParseSpecResult(error="Model returned no structured StrategySpec output.")

    try:
        validated = parsed if isinstance(parsed, StrategySpec) else StrategySpec.model_validate(parsed)
    except ValidationError as exc:
        return ParseSpecResult(error=f"Generated spec failed validation: {exc}")

    return ParseSpecResult(spec=validated)
