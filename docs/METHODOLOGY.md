# QuantCritic Quantitative Methodology

This document details the execution mechanics, accounting conventions, and statistical formulas implemented in QuantCritic.

---

## 1. Execution & Accounting Mechanics (`quantcritic/engine/`)

### Causal Order Flow
- Every strategy implements `generate(df, params)` using **only bars $\le t$** to emit a signal at the close of bar $t$.
- Orders enter at the **open of bar $t + d$** (default $d = 1$; $d = 2$ during the 1-bar delay stress test). Signals on the final bar have no $t + 1$ open and are ignored.
- One position per symbol is permitted at any time. At each global timestamp:
  1. Scheduled exits (`time_stop`, `exit_signal`, `opposite_signal`) execute first at the bar open.
  2. Flat symbols with valid entry signals (`signal != 0` and `stop_dist > 0`) enter at the bar open.
  3. Intraday protective stops and take-profits are checked from the entry bar onward.
- **Window evaluation (user-selected `[start, end]` range, validation, every walk-forward OOS window, locked test)**: signals are generated on all history up to the window end, so indicators and stateful strategies are warm when the window opens; the engine (`run_backtest(..., trade_start=...)`) only accepts entries at bars $\ge$ the window start and reports equity/metrics for the window only. A signal on the last bar before the window may therefore enter at the window's first open. Nothing after the window end is ever generated. When a custom `[start, end]` date range is selected, the 60% / 20% / 20% train / validation / locked-test split is applied inside that range while pre-`start` history may still warm up indicators.

### Fill Rules & Worst-Case Tie-Breaking
Let $\delta = \text{slippage\_bps} / 10000$ and $\text{dir} \in \{+1, -1\}$:
- **Entry Fill**: $P_{\text{entry}} = O_t \cdot (1 + \text{dir} \cdot \delta)$.
- **Protective Stop**: $P_{\text{stop}} = P_{\text{entry}} - \text{dir} \cdot \text{stop\_dist}$.
  - If a subsequent bar opens beyond the stop ($O_t \le P_{\text{stop}}$ for longs; $O_t \ge P_{\text{stop}}$ for shorts), the fill occurs at the **bar open** (gap-through-stop) plus adverse exit slippage.
  - If both stop and take-profit prices are touched inside the same bar's `[low, high]`, **stop is assumed hit first**.
- **Take-Profit Fill**: Conservative limit fill at $P_{\text{tp}}$ (with configured exit slippage).
- **Exit Fill**: $P_{\text{exit}} = P_{\text{raw\_exit}} \cdot (1 - \text{dir} \cdot \delta)$.

### Fees, Funding, and R-Multiples
- **Taker Fees**: Charged on both entry and exit notionals: $\text{fee} = q \cdot P \cdot r_{\text{fee}}$.
- **Perpetual Funding**: For every funding timestamp $\tau$ inside $(t_k, t_{k+1}]$ while a position is held across bar $k$ close, cashflow is:
  $$\text{funding\_pnl} = -\text{dir} \cdot (q \cdot C_k) \cdot r_{\text{fund}}(\tau)$$
  Positive funding reduces long PnL and increases short PnL. A settlement exactly at the entry open is not charged; one exactly at the exit open is. A position stopped out *inside* a bar is not charged for settlements inside that bar (exact on 1h and epoch-aligned 4h bars, whose boundaries coincide with the 00/08/16 UTC settlements; an approximation on 1d bars).
- **Strategy funding feature**: `funding_rate` on bar $t$ is the last settlement strictly before the bar close (`attach_causal_funding`).
- **Net PnL & R-Multiple**:
  $$\text{net\_pnl} = \text{dir} \cdot q \cdot (P_{\text{exit}} - P_{\text{entry}}) - \text{fees} + \text{funding\_pnl}$$
  $$R = \frac{\text{net\_pnl}}{\text{risk\_amount}}, \quad \text{risk\_amount} = q \cdot \text{stop\_dist}$$

### Position Sizing (`quantcritic/risk/sizing.py`)
- **Fixed-Fractional**: $q_{\text{raw}} = (\text{equity} \cdot r_{\text{risk}}) / \text{stop\_dist}$, capped so total portfolio gross notional never exceeds $\text{max\_leverage} \cdot \text{equity}$.
- **Capped Fractional Kelly**: $f^* = \alpha \cdot \max\left(0, p - \frac{1-p}{b}\right)$, clamped to $[0, f_{\max}]$, where $p$ is win rate and $b = \overline{R}_{\text{win}} / |\overline{R}_{\text{loss}}|$.
- **Risk of Ruin**: Exponential-barrier solution $\theta > 0$ to $p e^{-\theta a} + (1-p) e^{\theta d} = 1$, giving $P(\text{ruin}) \approx \exp(-\theta \cdot \text{DD}_{\text{ruin}} / r_{\text{risk}})$.

### Performance Ratios (`quantcritic/engine/metrics.py`)
Daily returns are computed from end-of-day mark-to-market equity (`UTC 1D`, day 0 anchored at the initial capital) and annualized using **365 days/year** for 24/7 crypto markets:
$$\text{Sharpe} = \frac{\bar{r}_d}{s_d} \sqrt{365}, \qquad \text{Sortino} = \frac{\bar{r}_d}{\sqrt{\frac{1}{N}\sum \min(r_d, 0)^2}} \sqrt{365}$$

---

## 2. Integrity & Robustness Tests (`quantcritic/audit/`)

### 2.1 Causal Truncation & Delay Tests (`lookahead.py`)
- **Truncation Test**: Draws $K = 25$ random cut points $t$ per symbol, calls `strategy.generate(df.iloc[:t+1])`, and verifies equality of every order field — `signal`, `stop_dist`, `tp_r`, `max_bars`, `exit_signal` — on the trailing 100 bars $\le t$ versus the full-sample run (floats compared to round-off, `rtol=1e-9`). Any discrepancy triggers an automatic `REJECT` verdict.
- **Delay Test**: Re-runs the backtest with `entry_delay_bars = 2`. An expectancy collapse $> 70\%$ raises an execution-timing `WARN`.

### 2.2 Chronological Splits & Rolling Walk-Forward (`walkforward.py`)
- Chronological partition: **60% Train / 20% Validation / 20% Locked Test**.
- Grid search optimizes Sharpe ratio (subject to minimum trade count) strictly on Train.
- Validation is evaluated as a window of Train+Validation history (signals warmed up on Train, only trades entered inside Validation count).
- Rolling walk-forward (12-month train $\rightarrow$ 3-month OOS trade, stepped every 3 months) runs strictly inside Train+Validation and stitches OOS equity. Fold parameter selection uses signals generated once on Train+Validation and sliced to each 12-month window; for any strategy that passes the truncation test this equals per-fold generation (a strategy that fails it is REJECTED anyway). Each fold's OOS window is evaluated with signals regenerated on history that ends exactly at that window's end.
- The 20% **Test slice is locked**: it is wrapped in a `LockedSlice` whose only accessor, `unlock()`, increments `access_count`; `final_test()` is the only caller and can run once. No optimization, Monte Carlo, or interactive experiment can inspect it before. The test-period result is a window run over the full history (warm signals, entries only from `test_start`).

### 2.3 Deflated Sharpe Ratio (`overfit.py`)
Following **Bailey & López de Prado (2014)**, given $N$ trials with cross-sectional variance $V[\{\widehat{SR}_n\}]$, daily non-annualized Sharpe $\widehat{SR}$, sample length $T$, skewness $\hat{\gamma}_3$, and Pearson kurtosis $\hat{\gamma}_4$:
$$SR_0 = \sqrt{V[\{\widehat{SR}_n\}]} \left((1 - \gamma)\Phi^{-1}\!\left(1 - \frac{1}{N}\right) + \gamma \Phi^{-1}\!\left(1 - \frac{1}{Ne}\right)\right)$$
$$\widehat{\sigma}_{SR} = \sqrt{\frac{1 - \hat{\gamma}_3 \widehat{SR} + \frac{\hat{\gamma}_4 - 1}{4}\widehat{SR}^2}{T - 1}}, \qquad \text{DSR} = \Phi\!\left(\frac{\widehat{SR} - SR_0}{\widehat{\sigma}_{SR}}\right)$$
where $\gamma \approx 0.5772156649$ is the Euler–Mascheroni constant.

### 2.4 Probability of Backtest Overfitting via CSCV (`overfit.py`)
Following **Bailey, Borwein, López de Prado & Zhu (2015)**, the daily return matrix $R \in \mathbb{R}^{T \times M}$ across all $M$ parameter combinations is partitioned into $S = 10$ contiguous chronological blocks. Across all $\binom{10}{5} = 252$ combinations assigning $S/2$ blocks to in-sample (IS) and $S/2$ to out-of-sample (OOS):
- Select $n^* = \arg\max_m \text{Sharpe}_{\text{IS}}(m)$.
- Measure relative rank $\bar{\omega}_{n^*} \in (0, 1)$ of strategy $n^*$ among all $M$ strategies OOS, and compute logit $\lambda_c = \ln(\bar{\omega}_{n^*} / (1 - \bar{\omega}_{n^*}))$.
- $\text{PBO} = \mathbb{P}(\lambda_c < 0)$. Pure-noise strategies yield $\text{PBO} \approx 0.50$.

### 2.5 Bootstrap & Matched Random-Entry Baseline (`montecarlo.py`)
- **Trade Bootstrap**: 1,000 resamples with replacement over observed trade $R$-multiples yield empirical distributions for total return, probability of loss, and 95th-percentile max drawdown.
- **Random-Entry Baseline**: 200 vectorized portfolios matching per-symbol trade count, empirical holding-time distribution (used as the time stop), long/short ratio, ATR stop rule (ATR known at the close of the bar before entry), TP rule, fees, slippage and perpetual funding (engine accrual rule) with randomized entry bars, so real and random $R$ are measured on identical terms. Finite-sample one-sided $p$-value:
  $$p = \frac{1 + \sum_{b=1}^{200} \mathbf{1}\{\bar{R}_{\text{rand}, b} \ge \bar{R}_{\text{real}}\}}{201}$$

### 2.6 Cost Stress, Regime Breakdown & Prop-Firm Simulation
- **Cost Stress (`costs.py`)**: 4$\times$4 grid (`fee` $\in \{0\times, 1\times, 2\times, 3\times\}$, `slippage` $\in \{0, 2, 5, 10\}$ bps) plus a linear-fit break-even one-way cost; `break_even_extrapolated` is set when the estimate lies beyond the largest cost tested (25 bps/side with default fees).
- **Regimes (`regime.py`)**: Causal daily BTC classification: `bull` ($\text{close} \ge \text{SMA}_{200}$ and $\sigma_{30\text{d}} \le q_{0.80}$), `bear` ($\text{close} < \text{SMA}_{200}$), `chop` ($\text{close} \ge \text{SMA}_{200}$ with $\sigma_{30\text{d}} > q_{0.80}$), where $q_{0.80}$ is the **expanding** 80th percentile of 30-day vol (no full-sample statistic), the SMA needs a full 200 days, and labels are lagged one day so a trade entered on day $d$ is tagged with data up to the close of $d-1$. Days without enough history are reported as `unclassified_regime_trades`. Plus calendar-year returns, symbol profit shares, long vs short stats, and removal of the best 5% of trades.
- **Prop-Firm Check (`propfirm.py`)**: Rolling weekly starts over two-step evaluation rules (+8% Phase 1, +5% Phase 2, 5% daily loss limit from day-start equity, 10% static max drawdown, minimum 5 active trading days).

### 2.7 Composite Scorecard (`scorecard.py`)
**Step 1 — weighted sub-scores (0–100).** Only out-of-sample evidence earns edge points; train-period performance earns nothing directly.

| Sub-score | Weight | Inputs |
|---|---|---|
| Integrity | 15% | truncation pass/fail, 1-bar delay expectancy collapse |
| OOS edge (pre-test) | 20% | 0.5·validation Sharpe + 0.5·rolling-WF Sharpe, validation expectancy, random-entry $p$ |
| Locked test | 20% | test Sharpe (60 pts) and test expectancy (40 pts); scaled by $n/10$ if fewer than 10 test trades |
| Robustness | 15% | DSR, $1-\text{PBO}$, neighbour stability ratio |
| Cost resilience | 10% | break-even bps, 2×fee + 5 bps stress expectancy |
| Consistency | 10% | positive years, profitable BTC regimes, survives top-5% removal (train+val) |
| Risk | 10% | full-period max DD, bootstrap p95 DD, bootstrap $P(\text{loss})$, risk of ruin from the real win rate and average win/loss $R$ |

**Step 2 — hard gates by `strictness` mode** (`"strict"`, `"balanced"` [default], `"relaxed"`). Hard gates cap both the score and the verdict so good in-sample or robustness numbers can never outvote failed integrity or out-of-sample checks. A period "loses" when its Sharpe, total return or expectancy is $\le 0$. In **all** modes, real metric numbers are always shown unchanged and truncation/lookahead failure triggers `REJECT` ($\text{score} \le 20$).

#### `strict` Mode (Original Gates; `PROMISING` $\ge 70$, `NEEDS WORK` $45\text{–}70$, `REJECT` $< 45$)
| Gate | Condition | Score cap | Verdict ceiling |
|---|---|---|---|
| `integrity` | truncation (lookahead) test failed | 20 | REJECT |
| `oos_fail` | locked test lost **and** validation or rolling WF lost | 40 | REJECT |
| `test_fail` | locked test lost | 55 | NEEDS WORK |
| `preoos_fail` | validation (Sharpe or expectancy) or rolling WF Sharpe $\le 0$ | 60 | NEEDS WORK |
| `promising_requirements` | fewer than 10 test trades, base-cost expectancy $\le 0$, random-entry $p > 0.10$, or DSR $< 0.50$ | 69.9 | NEEDS WORK |

#### `balanced` Mode (Default; `PROMISING` $\ge 65$, `NEEDS WORK` $45\text{–}65$, `REJECT` $< 45$)
| Gate | Condition | Score cap | Verdict ceiling |
|---|---|---|---|
| `integrity` | truncation (lookahead) test failed | 20 | REJECT |
| `oos_fail` | locked test lost **and** validation or rolling WF lost | 40 | REJECT |
| `test_fail` | locked test lost | 55 | NEEDS WORK |
| `preoos_fail` | validation or rolling WF lost **and** locked test profitable **and** fewer than 20 test trades (if $\ge 20$ test trades, allowed) | 60 | NEEDS WORK |
| `promising_requirements` | fewer than 10 test trades, base-cost expectancy $\le 0$, random-entry $p > 0.20$, or DSR $< 0.30$ | 64.9 | NEEDS WORK |

#### `relaxed` Mode (Exploratory; `PROMISING` $\ge 60$, `NEEDS WORK` $45\text{–}60$, `REJECT` $< 45$)
| Gate | Condition | Score cap | Verdict ceiling |
|---|---|---|---|
| `integrity` | truncation (lookahead) test failed | 20 | REJECT |
| `test_fail` | locked test lost | 55 | NEEDS WORK |
| `promising_requirements` | fewer than 5 test trades or base-cost expectancy $\le 0$ | 59.9 | NEEDS WORK |

**Step 3 — verdict** from the capped score using the active mode's `PROMISING` threshold ($70$ / $65$ / $60$) and `NEEDS WORK` floor ($45$). The identical strictness gates are applied by `AuditSession._apply_hard_gates` / `enforce_hard_gates` to any verdict submitted by the Claude AI critic (if downgraded, the gate reason is prepended to `red_flags`), as well as by the deterministic offline critic (`agent/offline.py`).

### 2.8 Causal No-Code Builder & Python Plugins (`strategies/builder.py`, `strategies/custom/`)
- **No-Code Builder (`StrategySpec`)**: Supports only causal closed-bar indicators (`close`, `open`, `high`, `low`, `ema`, `sma`, `rsi`, `atr`, `donchian_upper`/`donchian_lower` shifted by 1 bar, `bb_upper`/`bb_lower`/`bb_mid`, trailing `bb_bw_pct` on $0\text{–}100$ scale, `macd_line`/`macd_signal`/`macd_hist`, trailing `volume_z`, trailing `funding_z` on the causally attached `funding_rate` column, and `roc`) and operators (`crosses_above`, `crosses_below`, `greater_than`, `less_than`). Negative shifts, centered windows, and extra fields are rejected by strict Pydantic schemas (`extra="forbid"`). Entry signals emit single-bar transition events with warm-up bars zeroed; `exit_rules` close open positions at the next bar open; an `atr_stop_mult > 0` stop is mandatory.
- **Python Plugins (`quantcritic/strategies/custom/`)**: Subclasses of `Strategy` (excluding `template_strategy.py` and `_`-prefixed files) are auto-discovered and subject to the exact same causal truncation and robustness audit battery.
