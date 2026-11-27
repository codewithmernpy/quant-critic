"""Position sizing and risk-of-ruin models for crypto perpetual futures."""

from __future__ import annotations

import math

from scipy.optimize import brentq


def fixed_fractional_size(
    equity: float,
    entry_price: float,
    stop_dist: float,
    risk_per_trade: float = 0.005,
    max_leverage: float = 3.0,
    available_notional: float | None = None,
) -> tuple[float, float]:
    """Compute position quantity and actual dollar risk under a leverage cap.

    Returns (qty, risk_amount) where risk_amount = qty * stop_dist.
    """
    if equity <= 0.0 or entry_price <= 0.0 or stop_dist <= 0.0 or risk_per_trade <= 0.0:
        return 0.0, 0.0

    target_risk = equity * risk_per_trade
    raw_qty = target_risk / stop_dist
    max_notional = equity * max_leverage
    if available_notional is not None:
        max_notional = min(max_notional, max(0.0, available_notional))

    if max_notional <= 0.0:
        return 0.0, 0.0

    notional = raw_qty * entry_price
    if notional > max_notional:
        qty = max_notional / entry_price
    else:
        qty = raw_qty

    actual_risk = qty * stop_dist
    return float(qty), float(actual_risk)


def volatility_target_size(
    equity: float,
    price: float,
    realized_vol_ann: float,
    target_vol_ann: float = 0.15,
    max_leverage: float = 3.0,
    n_positions: int = 1,
) -> float:
    """Compute position quantity targeting a fixed annualized portfolio volatility."""
    if equity <= 0.0 or price <= 0.0 or realized_vol_ann <= 0.0 or target_vol_ann <= 0.0:
        return 0.0
    slots = max(1, n_positions)
    target_notional = (equity * (target_vol_ann / realized_vol_ann)) / slots
    capped_notional = min(target_notional, equity * max_leverage)
    return float(capped_notional / price)


def capped_fractional_kelly(
    win_rate: float,
    avg_win_r: float,
    avg_loss_r: float = 1.0,
    fraction: float = 0.25,
    max_risk: float = 0.02,
) -> float:
    """Compute capped fractional Kelly risk-per-trade from trade R-multiple stats."""
    if not (0.0 < win_rate < 1.0) or avg_win_r <= 0.0 or abs(avg_loss_r) <= 0.0:
        return 0.0
    payoff_ratio = avg_win_r / abs(avg_loss_r)
    full_kelly = win_rate - (1.0 - win_rate) / payoff_ratio
    if full_kelly <= 0.0:
        return 0.0
    return float(min(max_risk, max(0.0, full_kelly * fraction)))


def estimate_risk_of_ruin(
    win_rate: float,
    avg_win_r: float,
    avg_loss_r: float = 1.0,
    risk_per_trade: float = 0.005,
    ruin_drawdown: float = 0.50,
) -> float:
    """Estimate probability of hitting `ruin_drawdown` before recovering, via exponential barrier.

    Solves p * exp(-theta * a) + (1 - p) * exp(theta * d) = 1 for theta > 0,
    yielding P(ruin) ~= exp(-theta * N_units).
    """
    if risk_per_trade <= 0.0 or ruin_drawdown <= 0.0:
        return 0.0
    if win_rate <= 0.0:
        return 1.0
    if win_rate >= 1.0:
        return 0.0

    a = float(avg_win_r)
    d = float(abs(avg_loss_r))
    if a <= 0.0:
        return 1.0
    if d <= 0.0:
        return 0.0

    expectancy = win_rate * a - (1.0 - win_rate) * d
    if expectancy <= 0.0:
        return 1.0

    n_units = ruin_drawdown / risk_per_trade

    def f(theta: float) -> float:
        return win_rate * math.exp(-theta * a) + (1.0 - win_rate) * math.exp(theta * d) - 1.0

    hi = 1.0
    while f(hi) < 0.0 and hi < 100.0:
        hi *= 2.0
    if f(hi) <= 0.0:
        return 0.0

    theta = brentq(f, 1e-8, hi, xtol=1e-10)
    ror = math.exp(-theta * n_units)
    return float(min(1.0, max(0.0, ror)))


def drawdown_risk_multiplier(current_drawdown: float) -> tuple[float, str]:
    """Recommend position-size scaling multiplier based on current account drawdown."""
    dd = abs(float(current_drawdown))
    if dd >= 0.12:
        return 0.25, "Severe drawdown (>=12%): cut risk to 25% of baseline and pause new risk."
    if dd >= 0.08:
        return 0.50, "Elevated drawdown (8-12%): halve risk per trade until recovery."
    if dd >= 0.04:
        return 0.75, "Moderate drawdown (4-8%): reduce risk per trade by 25%."
    return 1.00, "Drawdown within normal operating band (<4%): full baseline sizing permitted."
