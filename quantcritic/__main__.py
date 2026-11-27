"""Command-line interface for QuantCritic."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import urllib.error
from pathlib import Path
from typing import Sequence

from quantcritic.agent.critic import run_ai_critic
from quantcritic.agent.tools import AuditSession
from quantcritic.audit.walkforward import run_window_backtest
from quantcritic.config import (
    DEFAULT_MODEL,
    DEFAULT_STRICTNESS,
    DEFAULT_SYMBOLS,
    REPORTS_DIR,
    STRICTNESS_MODES,
    SUPPORTED_INTERVALS,
    AuditConfig,
    BacktestConfig,
    configure_logging,
    load_env_file,
)
from quantcritic.data.binance import (
    _month_range,
    _norm_month_str,
    download_symbol_data,
    ensure_symbol_data,
    get_stored_months,
    latest_complete_month,
    normalize_symbol,
)
from quantcritic.data.loader import (
    MarketData,
    load_market_data,
    parse_utc_bound,
    prepare_range_data,
    validate_date_range,
)
from quantcritic.engine.backtest import run_backtest
from quantcritic.report.html import generate_audit_reports, generate_leaderboard_report
from quantcritic.strategies.registry import get_strategy, list_strategies

logger = logging.getLogger("quantcritic.cli")
_CLI_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantcritic",
        description="QuantCritic — Realistic crypto-perpetuals backtester, robustness auditor, and AI strategy critic.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    sub = parser.add_subparsers(dest="command", required=True)

    # download
    p_dl = sub.add_parser("download", help="Download monthly klines and funding rates from data.binance.vision.")
    p_dl.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    p_dl.add_argument("--interval", default="1h", choices=list(SUPPORTED_INTERVALS))
    p_dl.add_argument("--start", default="2021-01", help="Start month YYYY-MM")
    p_dl.add_argument("--end", default="2025-10", help="End month YYYY-MM")
    p_dl.add_argument("--no-funding", action="store_true")
    p_dl.add_argument("--output-dir", type=Path, default=None)

    # backtest
    p_bt = sub.add_parser("backtest", help="Run a multi-symbol portfolio backtest.")
    p_bt.add_argument("--strategy", required=True, choices=list_strategies())
    p_bt.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    p_bt.add_argument("--interval", default="4h", choices=list(SUPPORTED_INTERVALS))
    p_bt.add_argument("--start", default=None, help="Backtest start date YYYY-MM-DD")
    p_bt.add_argument("--end", default=None, help="Backtest end date YYYY-MM-DD")
    p_bt.add_argument("--no-download", action="store_true", help="Forbid network downloads if data is missing.")
    p_bt.add_argument("--capital", type=float, default=10_000.0)
    p_bt.add_argument("--risk", type=float, default=0.005)
    p_bt.add_argument("--max-leverage", type=float, default=3.0)
    p_bt.add_argument("--fee", type=float, default=0.0005)
    p_bt.add_argument("--slippage", type=float, default=2.0)
    p_bt.add_argument("--params", default="{}", help="JSON parameter overrides")

    # audit
    p_au = sub.add_parser("audit", help="Run full robustness audit + AI/offline critic and write HTML+JSON reports.")
    p_au.add_argument("--strategy", required=True, choices=list_strategies())
    p_au.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    p_au.add_argument("--interval", default="4h", choices=list(SUPPORTED_INTERVALS))
    p_au.add_argument("--start", default=None, help="Audit start date YYYY-MM-DD")
    p_au.add_argument("--end", default=None, help="Audit end date YYYY-MM-DD")
    p_au.add_argument("--no-download", action="store_true", help="Forbid network downloads if data is missing.")
    p_au.add_argument("--capital", type=float, default=10_000.0)
    p_au.add_argument("--risk", type=float, default=0.005)
    p_au.add_argument("--max-leverage", type=float, default=3.0)
    p_au.add_argument("--fee", type=float, default=0.0005)
    p_au.add_argument("--slippage", type=float, default=2.0)
    p_au.add_argument("--ai", action="store_true", help="Use Claude AI critic (falls back offline if no key).")
    p_au.add_argument("--model", default=DEFAULT_MODEL)
    p_au.add_argument(
        "--strictness",
        default=DEFAULT_STRICTNESS,
        choices=list(STRICTNESS_MODES),
        help="Verdict strictness mode (default: balanced).",
    )
    p_au.add_argument("--output-dir", type=Path, default=REPORTS_DIR)

    # compare
    p_cmp = sub.add_parser("compare", help="Audit all built-in strategies (offline) and write leaderboard.html.")
    p_cmp.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    p_cmp.add_argument("--interval", default="4h", choices=list(SUPPORTED_INTERVALS))
    p_cmp.add_argument("--start", default=None, help="Comparison start date YYYY-MM-DD")
    p_cmp.add_argument("--end", default=None, help="Comparison end date YYYY-MM-DD")
    p_cmp.add_argument("--no-download", action="store_true", help="Forbid network downloads if data is missing.")
    p_cmp.add_argument("--exclude-leaky", action="store_true")
    p_cmp.add_argument(
        "--strictness",
        default=DEFAULT_STRICTNESS,
        choices=list(STRICTNESS_MODES),
        help="Verdict strictness mode (default: balanced).",
    )
    p_cmp.add_argument("--output-dir", type=Path, default=REPORTS_DIR)

    return parser


def _normalize_cli_symbols(raw_symbols: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in raw_symbols:
        for tok in re.split(r"[,\s]+", item.strip()):
            if not tok:
                continue
            norm = normalize_symbol(tok)
            if norm not in seen:
                seen.add(norm)
                out.append(norm)
    if not out:
        raise ValueError("At least one symbol must be provided.")
    return out


def _validate_cli_dates(start: str | None, end: str | None) -> tuple[str | None, str | None]:
    for label, val in (("--start", start), ("--end", end)):
        if val is not None:
            if not _CLI_DATE_RE.match(val.strip()):
                raise ValueError(f"Invalid {label} date {val!r}; expected YYYY-MM-DD.")
            parse_utc_bound(val.strip(), is_end=False)
    s_clean = start.strip() if start else None
    e_clean = end.strip() if end else None
    if s_clean and e_clean and s_clean > e_clean:
        raise ValueError(f"--start ({s_clean}) must be on or before --end ({e_clean}).")
    return s_clean, e_clean


def _load_cli_range_data(
    raw_symbols: Sequence[str],
    interval: str,
    start: str | None,
    end: str | None,
    no_download: bool,
    min_days: int,
    min_bars: int,
    label: str,
) -> tuple[MarketData, MarketData, MarketData | None, object]:
    symbols = _normalize_cli_symbols(raw_symbols)
    s_str, e_str = _validate_cli_dates(start, end)

    for sym in symbols:
        stored = get_stored_months(sym, interval="1h", check_sample=True)
        has_local = bool(stored["klines"])
        if not has_local and interval != "1h":
            stored_iv = get_stored_months(sym, interval=interval, check_sample=True)
            if stored_iv["klines"]:
                has_local = True
                stored = stored_iv
        if not has_local:
            if no_download:
                raise FileNotFoundError(
                    f"No local data found for {sym} and --no-download forbids network access."
                )
            dl_start = _norm_month_str(s_str) if s_str else "2021-01"
            dl_end = _norm_month_str(e_str) if e_str else latest_complete_month()
            res = ensure_symbol_data(sym, interval="1h", start_month=dl_start, end_month=dl_end)
            if res["status"] == "not_found":
                raise FileNotFoundError(res["message"])
            if res.get("listed_later"):
                print(f"[INFO] {res['message']}")
        elif (s_str or e_str) and not no_download:
            min_local = f"{min(stored['klines'])[0]:04d}-{min(stored['klines'])[1]:02d}"
            max_local = f"{max(stored['klines'])[0]:04d}-{max(stored['klines'])[1]:02d}"
            req_start_m = _norm_month_str(s_str) if s_str else min_local
            req_end_m = _norm_month_str(e_str) if e_str else max_local
            if req_start_m <= req_end_m:
                missing = [
                    m for m in _month_range(req_start_m, req_end_m)
                    if m not in stored["klines"] and m not in stored["known_unavailable"]
                ]
                if missing:
                    res = ensure_symbol_data(sym, interval="1h", start_month=req_start_m, end_month=req_end_m)
                    if res.get("listed_later"):
                        print(f"[INFO] {res['message']}")

    full_data = load_market_data(symbols, interval=interval)
    history_to_end, window_data, warmup_data, trade_start = prepare_range_data(
        full_data, start=s_str, end=e_str
    )
    validate_date_range(
        window_data,
        start_ts=parse_utc_bound(s_str, is_end=False),
        end_ts=parse_utc_bound(e_str, is_end=True),
        min_days=min_days,
        min_bars=min_bars,
        label=label,
    )
    return history_to_end, window_data, warmup_data, trade_start


def _cmd_download(args: argparse.Namespace) -> int:
    try:
        symbols = _normalize_cli_symbols(args.symbols)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    for sym in symbols:
        saved = download_symbol_data(
            symbol=sym,
            interval=args.interval,
            start_month=args.start,
            end_month=args.end,
            output_dir=args.output_dir,
            include_funding=not args.no_funding,
        )
        if not saved:
            print(f"Error: {sym} not found on Binance Vision for {args.start}..{args.end}.", file=sys.stderr)
            return 1
        print(f"{sym}: saved { {k: str(v) for k, v in saved.items()} }")
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    try:
        history_to_end, window_data, _, trade_start = _load_cli_range_data(
            raw_symbols=args.symbols,
            interval=args.interval,
            start=args.start,
            end=args.end,
            no_download=args.no_download,
            min_days=30,
            min_bars=1,
            label="Backtest",
        )
    except (ValueError, FileNotFoundError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    strat = get_strategy(args.strategy)
    params = strat.merge_params(json.loads(args.params))
    cfg = BacktestConfig(
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        max_leverage=args.max_leverage,
        fee_rate=args.fee,
        slippage_bps=args.slippage,
        interval=args.interval,
    )
    if trade_start is not None:
        res = run_window_backtest(history_to_end, strat, params, cfg, trade_start)
    else:
        res = run_backtest(history_to_end, strat, params, cfg)
    m = res.metrics
    eff_start = str(min(df.index[0] for df in window_data.ohlcv.values() if not df.empty))[:10]
    eff_end = str(max(df.index[-1] for df in window_data.ohlcv.values() if not df.empty))[:10]
    print(f"=== Backtest: {strat.name} ({', '.join(window_data.symbols)} @ {args.interval}) ===")
    print(f"Range        : {eff_start} to {eff_end} UTC")
    print(f"Final Equity : ${m['final_equity']:,.2f} ({m['total_return'] * 100:+.2f}%)")
    print(f"CAGR         : {m['cagr'] * 100:+.2f}% | Sharpe: {m['sharpe']:.2f} | Sortino: {m['sortino']:.2f}")
    print(f"Max Drawdown : {m['max_drawdown'] * 100:.2f}% ({m['max_drawdown_duration_days']:.1f} days)")
    print(
        f"Trades       : {m['n_trades']} | Win Rate: {m['win_rate'] * 100:.1f}% | "
        f"Avg R: {m['avg_r']:+.3f}R | PF: {m['profit_factor']:.2f}"
    )
    print(f"Total Fees   : ${m['total_fees']:,.2f} | Net Funding: ${m['total_funding']:+,.2f}")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    try:
        _, window_data, warmup_data, _ = _load_cli_range_data(
            raw_symbols=args.symbols,
            interval=args.interval,
            start=args.start,
            end=args.end,
            no_download=args.no_download,
            min_days=180,
            min_bars=30,
            label="Audit",
        )
    except (ValueError, FileNotFoundError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    strat = get_strategy(args.strategy)
    cfg = BacktestConfig(
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        max_leverage=args.max_leverage,
        fee_rate=args.fee,
        slippage_bps=args.slippage,
        interval=args.interval,
    )
    session = AuditSession(
        strategy=strat,
        data=window_data,
        config=cfg,
        audit_config=AuditConfig(strictness=args.strictness),
        warmup_data=warmup_data,
    )
    try:
        session.check_min_train_val_trades(min_trades=5)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    verdict = run_ai_critic(session, use_ai=args.ai, model=args.model)
    html_path, json_path = generate_audit_reports(session, output_dir=args.output_dir)

    sc = session.cached_results["scorecard"]
    sp = session.splits
    print(f"=== QuantCritic Audit: {strat.name} ===")
    print(f"Range        : {str(sp.train_start)[:10]} to {str(sp.test_end)[:10]} UTC")
    print(
        f"Splits       : Train {str(sp.train_start)[:10]}..{str(sp.train_end)[:10]} | "
        f"Val {str(sp.val_start)[:10]}..{str(sp.val_end)[:10]} | "
        f"Locked Test {str(sp.test_start)[:10]}..{str(sp.test_end)[:10]}"
    )
    print(f"Verdict mode : {args.strictness}")
    print(f"Verdict      : {verdict.verdict} ({verdict.score:.1f}/100) [source={verdict.source}]")
    print(f"Summary      : {verdict.summary}")
    print(f"Sub-Scores   : {sc['sub_scores']}")
    print(f"HTML Report  : {html_path}")
    print(f"JSON Report  : {json_path}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    try:
        _, window_data, warmup_data, _ = _load_cli_range_data(
            raw_symbols=args.symbols,
            interval=args.interval,
            start=args.start,
            end=args.end,
            no_download=args.no_download,
            min_days=180,
            min_bars=30,
            label="Compare",
        )
    except (ValueError, FileNotFoundError, urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    cfg = BacktestConfig(interval=args.interval)
    audit_cfg = AuditConfig(strictness=args.strictness)
    strat_names = list_strategies(include_leaky=not args.exclude_leaky)
    rows: list[dict[str, object]] = []

    for name in strat_names:
        strat = get_strategy(name)
        session = AuditSession(
            strategy=strat,
            data=window_data,
            config=cfg,
            audit_config=audit_cfg,
            warmup_data=warmup_data,
        )
        verdict = run_ai_critic(session, use_ai=False)
        sc = session.cached_results["scorecard"]
        wf = session.cached_results["walkforward"]
        ov = session.cached_results["overfit"]
        mc = session.cached_results["montecarlo"]
        cs = session.cached_results["costs"]
        la = session.cached_results["lookahead"]
        ft = session.cached_results["final_test"]
        full_m = session.cached_results["full_result"].metrics
        rows.append(
            {
                "strategy": name,
                "verdict": verdict.verdict,
                "score": float(verdict.score),
                "lookahead_status": la["status"],
                "val_sharpe": float(wf["validation_metrics"]["sharpe"]),
                "test_sharpe": float(ft["test_metrics"]["sharpe"]),
                "dsr": float(ov["dsr"]["dsr"]),
                "pbo": float(ov["pbo"]["pbo"]),
                "p_value": float(mc["random_baseline"]["p_value"]),
                "break_even_bps": float(cs["break_even_cost_bps"]),
                "full_return": float(full_m["total_return"]),
                "sub_scores": sc["sub_scores"],
            }
        )

    eff_start = str(min(df.index[0] for df in window_data.ohlcv.values() if not df.empty))[:10]
    eff_end = str(max(df.index[-1] for df in window_data.ohlcv.values() if not df.empty))[:10]
    html_path, json_path = generate_leaderboard_report(
        rows=rows,
        symbols=window_data.symbols,
        interval=args.interval,
        output_dir=args.output_dir,
        date_range=(eff_start, eff_end),
        strictness=args.strictness,
    )
    print(f"=== QuantCritic Leaderboard ({eff_start} to {eff_end} UTC | Verdict mode: {args.strictness}) ===")
    for r in sorted(rows, key=lambda x: float(x["score"]), reverse=True):
        print(
            f"  {str(r['strategy']):20s} {str(r['verdict']):11s} score={float(r['score']):5.1f} "
            f"lookahead={str(r['lookahead_status']):4s} val_sh={float(r['val_sharpe']):+.2f} "
            f"test_sh={float(r['test_sharpe']):+.2f} PBO={float(r['pbo']):.2f}"
        )
    print(f"Leaderboard HTML : {html_path}")
    print(f"Leaderboard JSON : {json_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    load_env_file()
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    if args.command == "download":
        return _cmd_download(args)
    if args.command == "backtest":
        return _cmd_backtest(args)
    if args.command == "audit":
        return _cmd_audit(args)
    if args.command == "compare":
        return _cmd_compare(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
