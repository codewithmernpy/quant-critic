"""Configuration models, constants, and path helpers for QuantCritic."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
SAMPLE_DATA_DIR: Path = DATA_DIR / "sample"
CACHE_DATA_DIR: Path = DATA_DIR / "cache"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
USER_STRATEGIES_DIR: Path = PROJECT_ROOT / "user_strategies"
USER_STRATEGY_EXAMPLES_DIR: Path = USER_STRATEGIES_DIR / "examples"
CUSTOM_STRATEGIES_DIR: Path = PROJECT_ROOT / "quantcritic" / "strategies" / "custom"

DEFAULT_SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
SUPPORTED_INTERVALS: tuple[str, ...] = ("1h", "4h", "1d")

INTERVAL_TIMEDELTA_HOURS: dict[str, int] = {
    "1h": 1,
    "4h": 4,
    "1d": 24,
}

PANDAS_RESAMPLE_MAP: dict[str, str] = {
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}

DEFAULT_MODEL: str = "claude-opus-5-5"
STRICTNESS_MODES: tuple[str, ...] = ("strict", "balanced", "relaxed")
DEFAULT_STRICTNESS: Literal["strict", "balanced", "relaxed"] = "balanced"


def load_env_file(env_path: Path | None = None) -> None:
    """Load key=value pairs from a .env file into os.environ if not already set."""
    path = env_path or (PROJECT_ROOT / ".env")
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root quantcritic logger for CLI runs."""
    logger = logging.getLogger("quantcritic")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)


class BacktestConfig(BaseModel):
    """Configuration for the crypto-perpetuals portfolio backtest engine."""

    initial_capital: float = Field(default=10_000.0, gt=0.0)
    risk_per_trade: float = Field(default=0.005, gt=0.0, le=0.25)
    max_leverage: float = Field(default=3.0, gt=0.0, le=125.0)
    fee_rate: float = Field(default=0.0005, ge=0.0)
    slippage_bps: float = Field(default=2.0, ge=0.0)
    funding_enabled: bool = Field(default=True)
    interval: Literal["1h", "4h", "1d"] = Field(default="4h")
    entry_delay_bars: int = Field(default=1, ge=1, le=10)
    sizing_method: Literal["fixed_fractional", "vol_target", "kelly"] = Field(
        default="fixed_fractional"
    )
    maintenance_margin_rate: float = Field(default=0.005, gt=0.0, lt=0.5)


class SplitConfig(BaseModel):
    """Chronological train / validation / locked-test split proportions."""

    train_ratio: float = Field(default=0.60, gt=0.1, lt=0.9)
    val_ratio: float = Field(default=0.20, gt=0.05, lt=0.5)
    test_ratio: float = Field(default=0.20, gt=0.05, lt=0.5)

    def model_post_init(self, __context: object) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")


class PropFirmConfig(BaseModel):
    """Two-step prop-firm evaluation parameters."""

    step1_target: float = Field(default=0.08, gt=0.0)
    step2_target: float = Field(default=0.05, gt=0.0)
    max_daily_loss: float = Field(default=0.05, gt=0.0, lt=1.0)
    max_total_loss: float = Field(default=0.10, gt=0.0, lt=1.0)
    min_trading_days: int = Field(default=5, ge=1)
    initial_balance: float = Field(default=100_000.0, gt=0.0)


class AuditConfig(BaseModel):
    """Parameters controlling the robustness and integrity audit battery."""

    seed: int = Field(default=42)
    lookahead_cuts: int = Field(default=25, ge=5)
    lookahead_window: int = Field(default=100, ge=10)
    mc_simulations: int = Field(default=1000, ge=50)
    random_baseline_runs: int = Field(default=200, ge=20)
    cscv_blocks: int = Field(default=10, ge=4)
    min_train_trades: int = Field(default=30, ge=1)
    wf_train_months: int = Field(default=12, ge=2)
    wf_test_months: int = Field(default=3, ge=1)
    strictness: Literal["strict", "balanced", "relaxed"] = Field(default=DEFAULT_STRICTNESS)
