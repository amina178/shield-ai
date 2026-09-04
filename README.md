# Shield-AI — Autonomous Risk Guardrails & Hedging Agent

**Lablab.ai × Alpaca AI Trading Agents Hackathon** · Agent Builder track — The Internet of Agents

> Scoring is terminal account equity with no benchmark. In a five-day window a 30% drawdown is
> unrecoverable. Maximising final equity is therefore a **risk-management problem, not a prediction
> problem**. Shield-AI harvests a small, statistically-grounded premium and refuses to let the book
> take a loss it cannot earn back.

---

## The thesis

Most AI trading agents chase alpha. Hand a language model a brokerage key, tell it to pick winners,
and you have a leveraged guess: no covariance matrix, no tail estimate, no loss budget. The model is
confident, not calibrated — and confidence is not a risk framework.

Shield-AI inverts the design. **Deterministic mathematics defines the boundary; the language model
only ever operates inside it.** The same GARCH model that measures risk also generates the trading
signal, so risk control and P&L generation are not two bolted-together systems — they are one model
used twice.

### Why an LLM belongs in a quant strategy at all

**GARCH cannot read that the FDA decision is on Thursday.**

Selling volatility into a known catalyst is a classic way to destroy an account. Implied volatility
is elevated *because* the market knows something is coming, so the "edge" the model sees is not an
edge — it is compensation for a risk the model structurally cannot perceive. The News Sentiment
Agent holds a **veto**, never a vote: it proposes nothing, sizes nothing, and answers exactly one
question — is there an unresolved catalyst inside the life of this option?

---

## Strategy

The agent trades the **variance risk premium**: the persistent gap between option-implied volatility
and the volatility that actually materialises.

| Market state | Signal | Action |
|---|---|---|
| IV ≫ GARCH forecast | Options are rich | **Sell premium** — cash-secured put → income |
| IV ≪ GARCH forecast | Options are cheap | **Buy protection** — long put |
| VaR limit breached, or delta out of band | Risk event | **Hedge**, regardless of how rich the premium |

Order of precedence is the thesis in executable form: **a VaR breach outranks any income
opportunity.** `tests/test_agents.py` asserts this directly — an edge of +12 volatility points must
still produce a hedge, not a sale, when VaR is over the mandate.

### Tenor is set by the scoring window, not by convention

Textbook premium selling uses 30–45 DTE, which assumes holding for a month. Theta decay accelerates
into expiry, so over a five-day scoring window a 45-DTE contract surrenders only a sliver of its
premium while a 7–21 DTE contract surrenders most of it. The cost is gamma: short-dated deltas move
violently. **Shortening tenor and loosening the risk limit would compound two risks**, so
`var_limit_pct` was tightened to 2.0% in the same change that shortened the window.

### Sizing is solved for, never chosen

The question is never "how many contracts do I want" but "how many fit inside what is left of the
mandate". A position may consume at most **half** the remaining headroom to the VaR limit, so a
single trade can never take the book from compliant to breached — the agent must always retain the
ability to hedge after acting.

---

## Architecture

```
Alpaca Market Data ─┐
Alpaca News API ────┼─► Quant Engine  (NumPy · SciPy · statsmodels · arch)
Alpaca Option Chain ┘     VaR ×3 · CVaR · GARCH(1,1) · beta-weighted Delta
                              │
                              ▼
                        GUARDRAIL CHECK ──── within mandate ───► monitor
                              │ breach / edge
                              ▼
                     Cognitive Layer
                       News Sentiment Agent → VETO on event risk
                       Risk-Manager Agent   → instrument, strike, expiry, size
                              │
                              ▼
                     Policy Filter (deterministic, un-overridable)
                       options level · liquidity · coverage · cost · buying power
                              │
                              ▼
                     Alpaca MCP Server → place_option_order
                              │
                              ▼
                  decisions.jsonl + trades.jsonl  (append-only evidence)
```

Order execution runs through Alpaca's official MCP server, so the agent reaches the broker over an
**open protocol** rather than a hard-coded SDK binding — the same agent could be pointed at any
MCP-compliant venue without a rewrite. Bulk market data for the quant engine uses `alpaca-py`
directly, which is the split Alpaca's own engineering team recommended.

---

## The official competition run

The scoring window ran **Mon 31 Aug 09:30 ET → Fri 4 Sep 09:30 ET**, and the rules require a
clean paper account with no pre-existing positions. Shield-AI was developed and debugged on a
separate testing account; the account below was created fresh at $100,000 for the official run,
which means the measured window is **one session plus one overnight gap**.

That constraint suits the thesis rather than fighting it. With the window closing at Friday's
opening bell, essentially all of the remaining risk is a single overnight gap — precisely the move
a delta-based model underestimates and precisely what a protective put exists to bound.

| | |
|---|---|
| Starting equity | $100,000.00 |
| Book | 6 inverse-volatility weighted equity positions, 54.2% invested |
| Portfolio VaR at seeding | 1.142% ($1,142) against a 2.00% mandate |
| Net beta-weighted delta | 0.431, inside the 0.30–0.85 band |
| Hedge placed | 3× `NVDA260911P00220000` @ 1.19 — order `acbd143b-46d6-4727-bc68-ab2feda45837` |

The hedge fired because GARCH forecast 42.2% realised volatility on NVDA while the options implied
only 34.1%: protection was eight volatility points cheap.

Three names were refused in the same cycle, each with its reason written to the log:

- **MSFT** — edge −14.6 points, the most attractive signal of the cycle, blocked because open
  interest of 429 fell below the liquidity floor. An option you cannot exit in a crisis is not a
  hedge, and a rich signal does not change that.
- **XOM** — same gate, open interest 535.
- **AAPL, JPM** — edge inside the ±4 point no-trade band.
- **SPY** — no contract close enough to the target delta for the reference IV to be meaningful.

The liquidity threshold was set before the run and deliberately **not** relaxed once it began
blocking attractive trades. Loosening a risk parameter after seeing it refuse a trade you wanted is
the exact failure this project is built to prevent.

## Evidence

A trading agent that cannot prove what it did is a story, not a system. Three artifacts:

| Artifact | What it proves |
|---|---|
| `logs/decisions.jsonl` | Every decision **including refusals** — a hedge declined on a liquidity gate is direct evidence the guardrails work |
| `logs/trades.jsonl` | Live orders with the **broker-assigned order ID**, reconcilable against the Alpaca account |
| Tearsheet | Sharpe (excess of risk-free), Sortino (downside deviation only), max drawdown, profit factor |

A snapshot of the official run's logs is committed under [`docs/run-2026-09-03/`](docs/) so the
decision trail can be read without running anything.

A log containing only executed trades proves the agent traded. A log containing refusals proves the
agent had judgement.

**Win rate is never reported alone.** Premium selling wins often and small and loses rarely and
large, so a bare win rate flatters the strategy and tells the reader nothing without the loss
distribution.

---

## Verification — 46 tests

The risk model is not asserted, it is **statistically validated**. Each test constructs synthetic
data whose correct answer is known analytically, then checks the code reproduces it.

```bash
PYTHONPATH=. python3 tests/test_quant.py      # 13/13
PYTHONPATH=. python3 tests/test_execution.py  # 16/16
PYTHONPATH=. python3 tests/test_agents.py     # 17/17
```

Highlights:

- **Parametric VaR** reproduces the closed-form Gaussian quantile to 4 decimal places
- **Three estimators** (parametric, historical, Monte Carlo) agree to within 0.1% on Gaussian data
- **Beta-weighted delta** recovers a hand-constructed beta of exactly 2.0
- **Kupiec proportion-of-failures test** accepts a correctly calibrated model (p = 0.20) and rejects
  an overconfident one (p < 1e-16)
- **Christoffersen independence test** catches clustered breaches that Kupiec passes — ten
  exceptions at exactly the right 1% rate, arranged in two blocks of five, give Kupiec p = 1.0000
  and Christoffersen p ≈ 0
- **GARCH** recovers the parameters of a simulated GARCH process
- **Naked short calls are rejected unconditionally** — no confidence level and no attractive premium
  can open that gate
- **The sentiment veto outranks every other gate**; if that test fails, the cognitive layer is
  decorative

### A bug worth documenting

The VaR estimators take normalised weights, so they return risk as a fraction of *invested
notional*. The mandate is a fraction of *account equity*. Without converting by gross leverage, a
half-invested book reported the same VaR as a fully invested one — meaning the position-sizing loop
could cut exposure forever without the reported risk ever falling.

Every formula was individually correct. The error lived at the seam, in a silent assumption that
"fraction" meant the same thing in two places. That class of bug is not caught by checking formulas,
only by checking invariants: **halve the book, halve the VaR.** It is now `test_var_scales_with_exposure`.

---

## Setup

Requires Python 3.10+ and Alpaca options **Level 3** (Level 2 for long puts, Level 3 for multi-leg).

```bash
pip install -r requirements.txt
PYTHONPATH=. python3 scripts/setup_env.py       # prompts for keys, verifies them
PYTHONPATH=. python3 scripts/preflight.py       # 8 integration checks
PYTHONPATH=. python3 scripts/seed_portfolio.py  # dry run; --execute to submit
PYTHONPATH=. python3 scripts/run_agent.py       # dry run; --execute, --loop N
```

Every check in `preflight.py` works with the market closed — historical bars, account state, news
and contract metadata are available 24/7, so the whole stack can be debugged outside session hours.

---

## Repository layout

```
shield_ai/
  config.py              the risk mandate — every number the agent may act on
  quant/
    risk.py              VaR (parametric/historical/Monte Carlo), CVaR, beta-weighted delta
    garch.py             GARCH(1,1) forecast + variance-risk-premium signal
    var_backtest.py      Kupiec & Christoffersen statistical validation
  data/
    alpaca_client.py     bars, account, positions, news, option chain and Greeks
  agents/
    sentiment.py         event-risk veto (LLM, with deterministic keyword fallback)
    risk_manager.py      decision tree, contract selection, risk-budget sizing
  execution/
    policy.py            liquidity, coverage, cost, buying-power and level gates
    audit.py             append-only decision + trade logs, tearsheet
scripts/
  setup_env.py           credential setup with live verification
  preflight.py           8 integration checks, all runnable market-closed
  seed_portfolio.py      inverse-volatility base book, scaled to the mandate
  run_agent.py           the agent cycle
tests/                   46 verification tests
```

---

## Design notes

**Three VaR estimators, and the mandate uses the maximum.** If they disagree materially the
distributional assumption is doing heavy lifting, and that disagreement is itself a risk signal
(exposed as `model_disagreement`). Averaging would let an optimistic model dilute a pessimistic
one — exactly backwards for a risk system.

**Student-t with 5 degrees of freedom, not Gaussian.** Equity returns have fat tails; the normal
distribution systematically understates crash risk.

**Contracts are selected by delta, not by distance from the strike.** A 20-delta option means
roughly a 20% chance of finishing in the money whether the underlying is a utility or a
semiconductor. "5% out of the money" means something completely different on each.

**Limit orders at the natural midpoint.** On a $2.00/$2.20 quote, paying the ask costs $10 per
contract. Over a week of premium selling that is a meaningful share of the edge, and MCP supports
limit orders.

**The cognitive layer is not on the critical path.** Any LLM failure — network, rate limit, malformed
JSON — falls back to the deterministic keyword scanner. A trading agent that stops trading because
an LLM endpoint is slow has confused its cognitive layer with its critical path. Likewise, a failed
cycle in `--loop` mode is logged and skipped rather than fatal: a system that dies on a transient API
error stops managing the risk it already carries.

---

## License

MIT — see [LICENSE](LICENSE).
