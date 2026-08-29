# Shield-AI — Autonomous Risk Guardrails & Hedging Agent

**Lablab.ai × Alpaca AI Trading Agents Hackathon** · Agent Builder track — The Internet of Agents

> Scoring is terminal account equity with no benchmark. In a six-day window a 30% drawdown is
> unrecoverable. Maximising final equity is therefore a risk-management problem, not a prediction
> problem. Shield-AI harvests a small, statistically-grounded premium every day and refuses to let
> the book take a loss it cannot earn back.

---

## The idea in one paragraph

Most AI trading agents chase alpha: hand an LLM a brokerage key, tell it to pick winners, and you
have a leveraged guess with no covariance matrix, no tail estimate and no loss budget. Shield-AI
inverts the design. A deterministic quantitative engine defines a hard risk boundary; the language
model is only ever allowed to operate inside it. The same GARCH model that measures risk also
generates the trading signal, so risk control and P&L generation are not two bolted-together
systems — they are one model used twice.

## Strategy

The agent trades the **variance risk premium**: the persistent gap between option-implied
volatility and the volatility that actually materialises.

| Market state | Signal | Action |
|---|---|---|
| IV ≫ GARCH forecast | Options are rich | **Sell premium** — covered call / cash-secured put → income |
| IV ≪ GARCH forecast | Options are cheap | **Buy protection** — long put |
| VaR limit breached, or news shock | Risk event | **Hedge** regardless of IV |

Position size is never chosen by the model. It is solved for, so that the marginal contribution to
99% one-day Portfolio VaR stays inside the mandate defined in [`shield_ai/config.py`](shield_ai/config.py).

### Why an LLM belongs in a quant strategy at all

**GARCH cannot read that the FDA decision is on Thursday.** Selling volatility into a known
catalyst is a classic way to blow up an account, and a volatility model sees only numbers — it has
no view of the event calendar buried in unstructured news. The News Sentiment Agent parses the
Alpaca news stream and holds a **veto** over premium selling when event risk is present. That is
the honest justification for the cognitive layer: it supplies the information the mathematics
structurally cannot see.

## Architecture

```
Alpaca Market Data ─┐
Alpaca News API ────┼─► Quant Engine (NumPy/SciPy/statsmodels/arch)
Alpaca Option Chain ┘         │  VaR · CVaR · GARCH · beta-weighted Delta
                              ▼
                        GUARDRAIL CHECK ── within mandate ─► monitor
                              │ breach
                              ▼
                     Cognitive Layer (LLM)
                       News Sentiment Agent → veto / severity
                       Risk-Manager Agent  → instrument, strike, expiry, size
                              │
                              ▼
                     Policy & Constraint Filter  (deterministic)
                              │
                              ▼
                   Alpaca MCP Server  →  place_option_order
                              │
                              ▼
                Streamlit dashboard + immutable audit log
```

Execution runs exclusively through Alpaca's official MCP server, so the agent talks to the broker
over an open protocol rather than a hard-coded SDK binding. Market data for the quant engine uses
`alpaca-py` directly, which is the split Alpaca's own engineering team recommended.

## Verification

The risk model is not asserted — it is statistically validated. `tests/test_quant.py` builds
synthetic data with analytically known answers and checks the engine reproduces them, including:

- Parametric VaR against the closed-form Gaussian quantile
- Agreement between the parametric, historical and Monte Carlo estimators
- Recovery of a hand-constructed beta of exactly 2.0
- **Kupiec proportion-of-failures test** — accepts a correctly calibrated model, rejects an
  overconfident one
- **Christoffersen independence test** — detects clustered breaches that Kupiec alone passes
- GARCH parameter recovery from a simulated GARCH process

```bash
PYTHONPATH=. python3 tests/test_quant.py
# 12/12 passed
```

## Setup

Requires Python 3.10+ and **Alpaca options Level 3** on the paper account (Level 2 for long puts,
Level 3 for multi-leg collars).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in your paper keys
PYTHONPATH=. python3 tests/test_quant.py
```

## Repository layout

```
shield_ai/
  config.py            risk mandate — every number the agent may act on
  quant/
    risk.py            VaR (3 estimators), CVaR, beta-weighted delta
    garch.py           GARCH(1,1) forecast + IV-vs-forecast edge signal
    var_backtest.py    Kupiec & Christoffersen statistical validation
  data/                Alpaca market data + news ingestion
  agents/              news sentiment agent, risk-manager agent
  execution/           MCP client, policy filter, audit log
tests/
  test_quant.py        verification against analytically known answers
```

## Status

- [x] Quantitative risk engine — VaR, CVaR, beta-weighted Delta
- [x] GARCH(1,1) forecaster and variance-risk-premium signal
- [x] Kupiec / Christoffersen VaR backtests (12/12 verification tests passing)
- [ ] Alpaca data ingestion and portfolio seeding
- [ ] News Sentiment Agent (event-risk veto)
- [ ] Risk-Manager Agent and deterministic policy filter
- [ ] MCP execution gateway
- [ ] Streamlit dashboard

## License

MIT — see [LICENSE](LICENSE).
