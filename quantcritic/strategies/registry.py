"""Strategy registry with built-ins, JSON StrategySpecs, and Python plugin auto-discovery."""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from pathlib import Path

from quantcritic.config import (
    CUSTOM_STRATEGIES_DIR as _DEFAULT_CUSTOM_DIR,
    USER_STRATEGIES_DIR as _DEFAULT_USER_DIR,
    USER_STRATEGY_EXAMPLES_DIR as _DEFAULT_EXAMPLES_DIR,
)
from quantcritic.strategies.base import Strategy
from quantcritic.strategies.bollinger_squeeze import BollingerSqueezeStrategy
from quantcritic.strategies.builder import load_all_spec_strategies
from quantcritic.strategies.donchian_breakout import DonchianBreakoutStrategy
from quantcritic.strategies.ema_trend import EmaTrendStrategy
from quantcritic.strategies.funding_contrarian import FundingContrarianStrategy
from quantcritic.strategies.leaky_demo import LeakyDemoStrategy
from quantcritic.strategies.rsi_reversion import RsiReversionStrategy

logger = logging.getLogger(__name__)

USER_STRATEGIES_DIR: Path = _DEFAULT_USER_DIR
USER_STRATEGY_EXAMPLES_DIR: Path = _DEFAULT_EXAMPLES_DIR
CUSTOM_STRATEGIES_DIR: Path = _DEFAULT_CUSTOM_DIR

_BUILTIN_STRATEGIES: tuple[Strategy, ...] = (
    EmaTrendStrategy(),
    DonchianBreakoutStrategy(),
    RsiReversionStrategy(),
    BollingerSqueezeStrategy(),
    FundingContrarianStrategy(),
    LeakyDemoStrategy(),
)
_BUILTIN_NAMES: set[str] = {s.name for s in _BUILTIN_STRATEGIES}

_REGISTRY: dict[str, Strategy] = {s.name: s for s in _BUILTIN_STRATEGIES}
_LAST_SNAPSHOT: tuple[tuple[str, int, int], ...] | None = None


def _directory_snapshot(
    user_dir: Path,
    examples_dir: Path,
    custom_dir: Path,
) -> tuple[tuple[str, int, int], ...]:
    items: list[tuple[str, int, int]] = []
    for root, pattern in (
        (examples_dir, "*.json"),
        (user_dir, "*.json"),
        (custom_dir, "*.py"),
    ):
        if not root.is_dir():
            continue
        for p in sorted(root.glob(pattern)):
            if pattern == "*.py" and (p.name.startswith("_") or p.name == "template_strategy.py"):
                continue
            try:
                st = p.stat()
                items.append((str(p.resolve()), int(st.st_mtime_ns), int(st.st_size)))
            except OSError:
                continue
    return tuple(items)


def _discover_python_plugins(
    custom_dir: Path,
    existing_keys: set[str],
) -> dict[str, Strategy]:
    """Import concrete Strategy subclasses from `custom_dir` while logging and skipping errors."""
    plugins: dict[str, Strategy] = {}
    if not custom_dir.is_dir():
        return plugins

    for py_path in sorted(custom_dir.glob("*.py")):
        if py_path.name.startswith("_") or py_path.name == "template_strategy.py":
            continue
        try:
            st = py_path.stat()
            mod_name = f"_qc_custom_plugin_{py_path.stem}_{st.st_mtime_ns}"
            spec = importlib.util.spec_from_file_location(mod_name, py_path)
            if spec is None or spec.loader is None:
                logger.warning("Skipping custom strategy plugin %s: unable to create module spec.", py_path)
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(mod_name, None)
                raise

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ != module.__name__:
                    continue
                if obj is Strategy or not issubclass(obj, Strategy) or inspect.isabstract(obj):
                    continue
                try:
                    inst = obj()
                except Exception as exc:
                    logger.warning("Skipping plugin class %s in %s: instantiation failed (%s).", obj.__name__, py_path, exc)
                    continue
                key = str(getattr(inst, "name", "") or "").strip().lower()
                if not key:
                    logger.warning("Skipping plugin class %s in %s: empty `name` attribute.", obj.__name__, py_path)
                    continue
                if key in existing_keys or key in plugins:
                    logger.warning(
                        "Skipping plugin class %s in %s: strategy name %r is already registered.",
                        obj.__name__,
                        py_path,
                        key,
                    )
                    continue
                inst.name = key
                plugins[key] = inst
        except Exception as exc:
            logger.warning("Skipping broken custom strategy plugin %s: %s", py_path, exc)
    return plugins


def reload_custom_strategies(
    user_dir: Path | None = None,
    examples_dir: Path | None = None,
    custom_dir: Path | None = None,
) -> dict[str, Strategy]:
    """Reload built-ins, JSON specs (`custom:<name>`), and Python plugins into the registry."""
    global _LAST_SNAPSHOT
    u_dir = user_dir or USER_STRATEGIES_DIR
    ex_dir = examples_dir or USER_STRATEGY_EXAMPLES_DIR
    c_dir = custom_dir or CUSTOM_STRATEGIES_DIR

    new_reg: dict[str, Strategy] = {s.name: s for s in _BUILTIN_STRATEGIES}
    for key, strat in load_all_spec_strategies(user_dir=u_dir, examples_dir=ex_dir).items():
        if key in new_reg:
            logger.warning("Skipping JSON strategy spec %r: conflicts with built-in name.", key)
            continue
        new_reg[key] = strat

    plugins = _discover_python_plugins(c_dir, existing_keys=set(new_reg.keys()))
    new_reg.update(plugins)

    _REGISTRY.clear()
    _REGISTRY.update(new_reg)
    _LAST_SNAPSHOT = _directory_snapshot(u_dir, ex_dir, c_dir)
    return dict(_REGISTRY)


def _ensure_fresh_registry() -> None:
    snap = _directory_snapshot(USER_STRATEGIES_DIR, USER_STRATEGY_EXAMPLES_DIR, CUSTOM_STRATEGIES_DIR)
    if _LAST_SNAPSHOT != snap:
        reload_custom_strategies()


def register_strategy(strategy: Strategy) -> None:
    """Register an explicit Strategy instance in-memory (primarily for tests)."""
    key = strategy.name.strip().lower()
    if not key:
        raise ValueError("Strategy must define a non-empty `name`.")
    _REGISTRY[key] = strategy


def get_strategy(name: str) -> Strategy:
    """Return a registered Strategy instance by name."""
    _ensure_fresh_registry()
    key = name.strip().lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown strategy {name!r}. Available: {available}")
    return _REGISTRY[key]


def list_strategies(include_leaky: bool = True, include_custom: bool = True) -> list[str]:
    """List registered strategy names."""
    _ensure_fresh_registry()
    names = list(_REGISTRY.keys())
    if not include_leaky:
        names = [n for n in names if n != "leaky_demo"]
    if not include_custom:
        names = [n for n in names if n in _BUILTIN_NAMES]
    return names


# Initial discovery on module import
reload_custom_strategies()
