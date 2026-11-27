"""QuantCritic — Most crypto backtests lie. QuantCritic finds out how."""

from quantcritic.config import (
    AuditConfig,
    BacktestConfig,
    PropFirmConfig,
    SplitConfig,
    load_env_file,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AuditConfig",
    "BacktestConfig",
    "PropFirmConfig",
    "SplitConfig",
    "load_env_file",
]
