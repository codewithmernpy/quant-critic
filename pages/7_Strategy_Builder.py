"""Page 7: No-Code Causal Strategy Builder (with optional Claude NL translation)."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from quantcritic.strategies.builder import (
    INDICATOR_LABELS,
    SUPPORTED_INDICATORS,
    SUPPORTED_OPERATORS,
    BuilderStrategy,
    Condition,
    IndicatorOperand,
    NumberOperand,
    StrategySpec,
    format_condition,
    load_all_spec_strategies,
    mirror_condition,
    parse_strategy_description,
    save_strategy_spec,
)
from quantcritic.strategies.registry import reload_custom_strategies
from quantcritic.ui import init_page, load_range_data, render_sidebar

init_page("Strategy Builder", icon="🧩")
ctx = render_sidebar()

start_str = ctx["start_date"].isoformat()
end_str = ctx["end_date"].isoformat()

st.title("7 · No-Code Strategy Builder")
st.caption(
    f"Selected Preview Universe: `{', '.join(ctx['symbols'])}` · Interval: `{ctx['interval']}` · "
    f"Range: `{start_str}` to `{end_str}` UTC · All indicators are strictly causal on closed bars."
)


def _set_operand_state(prefix: str, op: IndicatorOperand | NumberOperand) -> None:
    if isinstance(op, NumberOperand):
        st.session_state[f"{prefix}_rtype"] = "number"
        st.session_state[f"{prefix}_num"] = float(op.value)
        return
    st.session_state[f"{prefix}_rtype"] = "indicator"
    st.session_state[f"{prefix}_ind"] = op.indicator
    st.session_state[f"{prefix}_period"] = int(op.period)
    st.session_state[f"{prefix}_std_dev"] = float(op.std_dev)
    st.session_state[f"{prefix}_lookback"] = int(op.lookback)
    st.session_state[f"{prefix}_fast"] = int(op.fast_period)
    st.session_state[f"{prefix}_slow"] = int(op.slow_period)
    st.session_state[f"{prefix}_sig"] = int(op.signal_period)


def _set_rule_state(prefix: str, cond: Condition) -> None:
    _set_operand_state(f"{prefix}_l", cond.left)
    st.session_state[f"{prefix}_op"] = cond.operator
    _set_operand_state(f"{prefix}_r", cond.right)


def _apply_spec_to_state(spec: StrategySpec) -> None:
    st.session_state["builder_name"] = spec.name
    st.session_state["builder_description"] = spec.description
    st.session_state["builder_combine"] = spec.combine
    st.session_state["builder_mirror"] = bool(spec.mirror_long_for_short)

    st.session_state["builder_n_long"] = len(spec.long_rules)
    for idx, cond in enumerate(spec.long_rules):
        _set_rule_state(f"builder_long_{idx}", cond)

    st.session_state["builder_n_short"] = len(spec.short_rules)
    for idx, cond in enumerate(spec.short_rules):
        _set_rule_state(f"builder_short_{idx}", cond)

    st.session_state["builder_atr_period"] = int(spec.exit.atr_period)
    st.session_state["builder_atr_stop_mult"] = float(spec.exit.atr_stop_mult)
    st.session_state["builder_use_tp"] = spec.exit.tp_r is not None
    st.session_state["builder_tp_r"] = float(spec.exit.tp_r) if spec.exit.tp_r is not None else 2.5
    st.session_state["builder_use_max_bars"] = spec.exit.max_bars is not None
    st.session_state["builder_max_bars"] = int(spec.exit.max_bars) if spec.exit.max_bars is not None else 24
    st.session_state["builder_exit_combine"] = spec.exit.exit_combine

    st.session_state["builder_n_exit"] = len(spec.exit.exit_rules)
    for idx, cond in enumerate(spec.exit.exit_rules):
        _set_rule_state(f"builder_exit_{idx}", cond)


def _ensure_defaults() -> None:
    if "builder_initialized" in st.session_state and "builder_name" in st.session_state:
        return
    default_spec = StrategySpec(
        name="custom_pullback",
        description="Close above EMA(200) and RSI(14) crosses above 40, mirrored for shorts.",
        long_rules=[
            Condition(
                left=IndicatorOperand(indicator="close"),
                operator="greater_than",
                right=IndicatorOperand(indicator="ema", period=200),
            ),
            Condition(
                left=IndicatorOperand(indicator="rsi", period=14),
                operator="crosses_above",
                right=NumberOperand(value=40.0),
            ),
        ],
        mirror_long_for_short=True,
        combine="all",
        exit={
            "atr_period": 14,
            "atr_stop_mult": 2.0,
            "tp_r": 2.5,
            "max_bars": 36,
        },
    )
    _apply_spec_to_state(default_spec)
    st.session_state["builder_initialized"] = True


_ensure_defaults()

# ------------------------------------------------------------------
# 1. Optional Plain-English -> StrategySpec with Claude
# ------------------------------------------------------------------
with st.expander("Describe Your Strategy in Plain English (Optional · Claude AI)", expanded=False):
    effective_key = (ctx["api_key"] or os.getenv("ANTHROPIC_API_KEY", "")).strip()
    if not effective_key:
        st.info(
            "No Anthropic API key configured. Enter a session key in the sidebar or set "
            "`ANTHROPIC_API_KEY` to draft specs from plain English. The manual rule builder below "
            "always works 100% locally without a key."
        )

    nl_desc = st.text_area(
        "Describe your strategy",
        placeholder=(
            "Example: Enter long when close is above EMA(200) and RSI(14) crosses above 40, "
            "mirror for shorts, use a 2x ATR(14) protective stop and 2.5R take profit."
        ),
        key="builder_nl_desc",
    )
    gen_disabled = not bool(effective_key)
    if st.button("Generate Draft Spec with Claude", disabled=gen_disabled, key="btn_nl_parse"):
        with st.spinner("Translating plain-English description into causal StrategySpec..."):
            res = parse_strategy_description(
                nl_desc,
                api_key=effective_key,
                model=ctx["model"],
            )
        if res.refusal:
            st.warning(res.error or "Model declined to parse this strategy description.")
        elif res.error:
            st.error(res.error)
        elif res.spec is not None:
            _apply_spec_to_state(res.spec)
            st.session_state["builder_ai_feedback"] = {
                "notes": res.spec.notes,
                "unsupported": list(res.spec.unsupported),
            }
            st.rerun()

    fb = st.session_state.get("builder_ai_feedback")
    if fb:
        st.success(
            "Draft loaded into the form below. Review and edit every rule and stop before saving (never auto-saved or auto-run)."
        )
        if fb.get("notes"):
            st.info(f"**Parser Notes:** {fb['notes']}")
        if fb.get("unsupported"):
            st.warning(
                "**Unsupported / Omitted Elements:**\n"
                + "\n".join(f"- {item}" for item in fb["unsupported"])
            )

# ------------------------------------------------------------------
# 2. Load Existing Saved or Example Spec
# ------------------------------------------------------------------
existing_specs = load_all_spec_strategies()
if existing_specs:
    ex_col1, ex_col2 = st.columns([3, 1])
    ex_names = sorted(existing_specs.keys())
    chosen_tmpl = ex_col1.selectbox("Load Example or Saved Spec into Form", options=ex_names, key="builder_load_sel")
    if ex_col2.button("Load Into Form", key="btn_load_tmpl"):
        _apply_spec_to_state(existing_specs[chosen_tmpl].spec)
        st.session_state.pop("builder_ai_feedback", None)
        st.rerun()

st.divider()

# ------------------------------------------------------------------
# 3. Manual Rule Builder Form
# ------------------------------------------------------------------
def _render_indicator_inputs(prefix: str, default_ind: str = "close") -> dict[str, Any]:
    ind_key = f"{prefix}_ind"
    if ind_key not in st.session_state:
        st.session_state[ind_key] = default_ind
    ind = st.selectbox(
        "Indicator",
        options=list(SUPPORTED_INDICATORS),
        format_func=lambda k: INDICATOR_LABELS.get(k, k),
        key=ind_key,
    )
    payload: dict[str, Any] = {"kind": "indicator", "indicator": ind}
    if ind in ("close", "open", "high", "low"):
        st.caption("Closed-bar price")
    elif ind in ("ema", "sma", "rsi", "atr", "donchian_upper", "donchian_lower", "volume_z", "funding_z", "roc"):
        default_p = 14 if ind != "funding_z" else 90
        if f"{prefix}_period" not in st.session_state:
            st.session_state[f"{prefix}_period"] = default_p
        min_p = 2 if ind in ("rsi", "volume_z", "funding_z") else 1
        st.session_state[f"{prefix}_period"] = max(min_p, int(st.session_state[f"{prefix}_period"]))
        payload["period"] = int(
            st.number_input("Period", min_value=min_p, max_value=1000, step=1, key=f"{prefix}_period")
        )
    elif ind in ("bb_upper", "bb_lower", "bb_mid"):
        if f"{prefix}_period" not in st.session_state:
            st.session_state[f"{prefix}_period"] = 20
        st.session_state[f"{prefix}_period"] = max(2, int(st.session_state[f"{prefix}_period"]))
        c_a, c_b = st.columns(2)
        payload["period"] = int(
            c_a.number_input("Period", min_value=2, max_value=500, step=1, key=f"{prefix}_period")
        )
        if ind != "bb_mid":
            if f"{prefix}_std_dev" not in st.session_state:
                st.session_state[f"{prefix}_std_dev"] = 2.0
            payload["std_dev"] = float(
                c_b.number_input("Std Dev (k)", min_value=0.1, max_value=10.0, step=0.25, key=f"{prefix}_std_dev")
            )
    elif ind == "bb_bw_pct":
        if f"{prefix}_period" not in st.session_state:
            st.session_state[f"{prefix}_period"] = 20
        if f"{prefix}_std_dev" not in st.session_state:
            st.session_state[f"{prefix}_std_dev"] = 2.0
        if f"{prefix}_lookback" not in st.session_state:
            st.session_state[f"{prefix}_lookback"] = 120
        st.session_state[f"{prefix}_period"] = max(2, int(st.session_state[f"{prefix}_period"]))
        c_a, c_b, c_c = st.columns(3)
        payload["period"] = int(c_a.number_input("BB Period", min_value=2, max_value=500, step=1, key=f"{prefix}_period"))
        payload["std_dev"] = float(c_b.number_input("Std (k)", min_value=0.1, max_value=10.0, step=0.25, key=f"{prefix}_std_dev"))
        payload["lookback"] = int(c_c.number_input("Lookback", min_value=5, max_value=2000, step=5, key=f"{prefix}_lookback"))
    elif ind in ("macd_line", "macd_signal", "macd_hist"):
        if f"{prefix}_fast" not in st.session_state:
            st.session_state[f"{prefix}_fast"] = 12
        if f"{prefix}_slow" not in st.session_state:
            st.session_state[f"{prefix}_slow"] = 26
        if f"{prefix}_sig" not in st.session_state:
            st.session_state[f"{prefix}_sig"] = 9
        c_a, c_b, c_c = st.columns(3)
        payload["fast_period"] = int(c_a.number_input("Fast", min_value=1, max_value=500, step=1, key=f"{prefix}_fast"))
        payload["slow_period"] = int(c_b.number_input("Slow", min_value=2, max_value=1000, step=1, key=f"{prefix}_slow"))
        payload["signal_period"] = int(c_c.number_input("Signal", min_value=1, max_value=500, step=1, key=f"{prefix}_sig"))
    return payload


def _render_rule_row(prefix: str, title: str) -> dict[str, Any]:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        col_l, col_op, col_rt, col_r = st.columns([3, 2, 1.5, 3])
        with col_l:
            left_dict = _render_indicator_inputs(f"{prefix}_l", default_ind="close")
        with col_op:
            if f"{prefix}_op" not in st.session_state:
                st.session_state[f"{prefix}_op"] = "greater_than"
            op = st.selectbox("Operator", options=list(SUPPORTED_OPERATORS), key=f"{prefix}_op")
        with col_rt:
            if f"{prefix}_r_rtype" not in st.session_state:
                st.session_state[f"{prefix}_r_rtype"] = "indicator"
            rtype = st.selectbox("Compare To", options=["indicator", "number"], key=f"{prefix}_r_rtype")
        with col_r:
            if rtype == "number":
                if f"{prefix}_r_num" not in st.session_state:
                    st.session_state[f"{prefix}_r_num"] = 0.0
                num_val = float(
                    st.number_input("Number Value", step=0.5, format="%.4f", key=f"{prefix}_r_num")
                )
                right_dict: dict[str, Any] = {"kind": "number", "value": num_val}
            else:
                right_dict = _render_indicator_inputs(f"{prefix}_r", default_ind="ema")
    return {"left": left_dict, "operator": op, "right": right_dict}


meta_c1, meta_c2 = st.columns([1, 2])
spec_name = meta_c1.text_input("Strategy Name (slug)", key="builder_name", help="Lowercase letters, digits, '_' or '-'.")
spec_desc = meta_c2.text_input("Description", key="builder_description")

opt_c1, opt_c2 = st.columns(2)
combine_mode = opt_c1.selectbox(
    "Combine Entry Rules",
    options=["all", "any"],
    format_func=lambda v: "all (AND — every rule must hold)" if v == "all" else "any (OR — any rule holds)",
    key="builder_combine",
)
mirror_short = opt_c2.checkbox("Mirror Long rules for Short entries", key="builder_mirror")

st.subheader("Long Entry Rules")
lc1, lc2, _ = st.columns([1, 1, 4])
if lc1.button("+ Add Long Rule", key="btn_add_long"):
    st.session_state["builder_n_long"] = int(st.session_state.get("builder_n_long", 1)) + 1
if lc2.button("- Remove Long Rule", key="btn_rem_long"):
    st.session_state["builder_n_long"] = max(0, int(st.session_state.get("builder_n_long", 1)) - 1)

n_long = int(st.session_state.get("builder_n_long", 1))
long_rules_raw = [_render_rule_row(f"builder_long_{i}", f"Long Rule #{i + 1}") for i in range(n_long)]

st.subheader("Short Entry Rules")
if mirror_short:
    st.caption("Automatically mirrored from Long Entry Rules:")
    for idx, r_raw in enumerate(long_rules_raw):
        try:
            cond_obj = Condition.model_validate(r_raw)
            st.code(f"Short #{idx + 1}: {format_condition(mirror_condition(cond_obj))}", language="text")
        except ValidationError:
            st.caption(f"Short #{idx + 1}: (waiting for valid Long Rule #{idx + 1})")
    short_rules_raw: list[dict[str, Any]] = []
else:
    sc1, sc2, _ = st.columns([1, 1, 4])
    if sc1.button("+ Add Short Rule", key="btn_add_short"):
        st.session_state["builder_n_short"] = int(st.session_state.get("builder_n_short", 0)) + 1
    if sc2.button("- Remove Short Rule", key="btn_rem_short"):
        st.session_state["builder_n_short"] = max(0, int(st.session_state.get("builder_n_short", 0)) - 1)
    n_short = int(st.session_state.get("builder_n_short", 0))
    short_rules_raw = [_render_rule_row(f"builder_short_{i}", f"Short Rule #{i + 1}") for i in range(n_short)]

st.subheader("Exits & Protective Stops")
ex1, ex2, ex3, ex4 = st.columns(4)
atr_period = int(ex1.number_input("ATR Period", min_value=1, max_value=500, step=1, key="builder_atr_period"))
atr_stop_mult = float(
    ex2.number_input(
        "ATR Stop Multiplier (> 0)",
        min_value=0.0,
        max_value=50.0,
        step=0.25,
        key="builder_atr_stop_mult",
    )
)
use_tp = ex3.checkbox("Take Profit (R)", key="builder_use_tp")
tp_r_val = float(ex3.number_input("TP Multiple (R)", min_value=0.1, max_value=100.0, step=0.25, key="builder_tp_r")) if use_tp else None
use_mb = ex4.checkbox("Max Bars Time Stop", key="builder_use_max_bars")
max_bars_val = int(ex4.number_input("Max Bars", min_value=1, max_value=10000, step=1, key="builder_max_bars")) if use_mb else None

st.markdown("**Optional Signal Exit Rules** *(close open position at next bar open)*")
ec1, ec2, ec3 = st.columns([1, 1, 3])
if ec1.button("+ Add Exit Rule", key="btn_add_exit"):
    st.session_state["builder_n_exit"] = int(st.session_state.get("builder_n_exit", 0)) + 1
if ec2.button("- Remove Exit Rule", key="btn_rem_exit"):
    st.session_state["builder_n_exit"] = max(0, int(st.session_state.get("builder_n_exit", 0)) - 1)
exit_combine = ec3.selectbox("Exit Rule Combine", options=["any", "all"], key="builder_exit_combine")
n_exit = int(st.session_state.get("builder_n_exit", 0))
exit_rules_raw = [_render_rule_row(f"builder_exit_{i}", f"Exit Rule #{i + 1}") for i in range(n_exit)]

# ------------------------------------------------------------------
# 4. Validate Spec & Live Signal Preview
# ------------------------------------------------------------------
raw_spec: dict[str, Any] = {
    "name": spec_name,
    "description": spec_desc,
    "long_rules": long_rules_raw,
    "short_rules": short_rules_raw,
    "mirror_long_for_short": mirror_short,
    "combine": combine_mode,
    "exit": {
        "atr_period": atr_period,
        "atr_stop_mult": atr_stop_mult,
        "tp_r": tp_r_val,
        "max_bars": max_bars_val,
        "exit_rules": exit_rules_raw,
        "exit_combine": exit_combine,
    },
}

st.divider()
try:
    spec = StrategySpec.model_validate(raw_spec)
except ValidationError as exc:
    st.error("Strategy specification has validation errors — fix the fields above before saving:")
    for err in exc.errors():
        loc = " -> ".join(str(x) for x in err.get("loc", ()))
        st.error(f"`{loc or 'spec'}`: {err.get('msg', 'invalid value')}")
    st.stop()

builder_strat = BuilderStrategy(spec)

st.subheader("Live Causal Signal Preview")
if ctx["date_error"]:
    st.error(ctx["date_error"])
else:
    try:
        history_to_end, _, _, trade_start = load_range_data(
            tuple(ctx["symbols"]),
            ctx["interval"],
            start_date=start_str,
            end_date=end_str,
        )
        preview_rows = []
        tot_long = 0
        tot_short = 0
        tot_exit = 0
        for sym, df in history_to_end.ohlcv.items():
            sig_df = builder_strat.generate(df)
            if trade_start is not None:
                sig_df = sig_df.loc[sig_df.index >= trade_start]
            n_l = int((sig_df["signal"] == 1).sum())
            n_s = int((sig_df["signal"] == -1).sum())
            n_e = int(sig_df["exit_signal"].sum())
            tot_long += n_l
            tot_short += n_s
            tot_exit += n_e
            preview_rows.append(
                {
                    "Symbol": sym,
                    "Window Bars": len(sig_df),
                    "Long Entries": n_l,
                    "Short Entries": n_s,
                    "Total Entries": n_l + n_s,
                    "Exit Triggers": n_e,
                }
            )
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Entry Signals", tot_long + tot_short)
        m2.metric("Long Entries", tot_long)
        m3.metric("Short Entries", tot_short)
        m4.metric("Exit Rule Triggers", tot_exit)
        m5.metric("Warm-up Bars", builder_strat.compute_warmup_bars())
        st.dataframe(pd.DataFrame(preview_rows), width="stretch", hide_index=True)
    except Exception as exc:
        st.error(f"Could not compute preview: {exc}")

with st.expander("Validated JSON Spec", expanded=False):
    st.code(spec.model_dump_json(indent=2), language="json")

# ------------------------------------------------------------------
# 5. Save & Jump to Backtest / Audit
# ------------------------------------------------------------------
def _save_and_activate(s: StrategySpec) -> str:
    saved_path = save_strategy_spec(s)
    reload_custom_strategies()
    reg_name = f"custom:{s.name}"
    st.session_state["_qc_pending_strategy"] = reg_name
    st.session_state["builder_saved_notice"] = (
        f"Saved `{reg_name}` to `{saved_path.as_posix()}`. It is now available in all strategy selectors and CLI (`--strategy {reg_name}`)."
    )
    return reg_name


b_save, b_bt, b_au = st.columns(3)
if b_save.button("Save Strategy", type="primary", key="btn_save_spec"):
    _save_and_activate(spec)
    st.rerun()

if st.session_state.get("builder_saved_notice"):
    st.success(st.session_state["builder_saved_notice"])

if b_bt.button("Save & Open Backtest", key="btn_save_and_backtest"):
    _save_and_activate(spec)
    st.session_state["_qc_autorun_backtest"] = True
    st.switch_page("pages/1_Backtest.py")

if b_au.button("Save & Open Audit", key="btn_save_and_audit"):
    _save_and_activate(spec)
    st.session_state["_qc_autorun_audit"] = True
    st.switch_page("pages/2_Audit.py")
