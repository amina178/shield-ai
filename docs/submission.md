# Shield-AI — Lablab.ai × Alpaca Hackathon Submission Copy

---

## 1. Title of Submission

**Shield-AI — Autonomous Risk Guardrails & Hedging Agent**

*Shorter alternative if the field is tight:* `Shield-AI: Autonomous Risk Guardrails Agent`

---

## 2. Short Description (255 char max)

```
Shield-AI is an autonomous risk agent for Alpaca. It computes 99% Portfolio VaR and net Delta continuously; when a guardrail breaks, its LLM layer reasons over the news shock and hedges the book with protective puts through the Alpaca MCP Server.
```

**246 characters** (verified) — 9 to spare under the 255 limit. ✅

---

## 3. Long Description

```
Most AI trading agents chase alpha. Shield-AI defends the downside.

Hand an LLM a brokerage key and tell it to pick winners, and you have a leveraged guess, not a quant system: no covariance matrix, no tail estimate, no loss budget. Shield-AI inverts that design. Deterministic mathematics defines the boundary; the language model only operates inside it.

A Python quant engine built on NumPy, SciPy and statsmodels continuously recomputes the paper portfolio's risk surface from Alpaca Market Data: a 99% one-day Portfolio Value-at-Risk (parametric, historical and Monte Carlo), Conditional VaR, GARCH(1,1) volatility forecasts, and net beta-weighted Delta. These are hard numbers, not model opinions.

The LLM stays asleep until a guardrail actually breaks. On breach, two agents wake: a News Sentiment Agent scores the severity, event type and persistence of the shock from the Alpaca News API, and a Risk-Manager Agent fuses that qualitative signal with the quantitative breach to select a hedge — instrument, strike, expiry and contract size. Every proposal passes a deterministic policy filter (maximum hedge cost, buying-power check, no naked short exposure) before it can reach the market.

Execution runs through Alpaca's official MCP Server, using get_option_chain and get_option_snapshot to price the chain, then place_option_order to establish a protective put or a multi-leg collar. Post-fill VaR is recomputed and written to an immutable audit log.

A Streamlit dashboard exposes live risk metrics alongside the agent's full reasoning trail, with human override on every decision.

Shield-AI treats risk management, not prediction, as the killer app for agentic finance.
```

**251 words** (verified) — comfortably over the 100-word minimum. ✅

---

## 4. Category

**Agent Builder track — The Internet of Agents**

*Justification to keep ready if judges ask why this track:*
Shield-AI is not a single prompt loop. It is a **multi-agent system where agents and infrastructure speak an open protocol**: a Sentiment Agent and a Risk-Manager Agent exchange structured signals, and both reach the broker exclusively through the Model Context Protocol rather than a hard-coded SDK binding. MCP is the interoperability layer — the same agent could be pointed at any MCP-compliant venue without a rewrite. That is precisely the Internet of Agents thesis: autonomous components negotiating over standard protocols instead of bespoke integrations.

---

## Verified technical facts (use these — they are from the official repo)

Official server: `alpacahq/alpaca-mcp-server` (v2, FastMCP-based). Python 3.10+. Env: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_PAPER_TRADE=true`.

Exact options tool names to cite in your README and demo:

| Tool | Use in Shield-AI |
|---|---|
| `get_option_chain` | Pull the full chain for the underlying being hedged |
| `get_option_snapshot` | Greeks + implied volatility for strike selection |
| `get_option_latest_quote` | Bid/ask for limit-price construction |
| `get_option_contracts` | Filter contracts by expiry / strike range |
| `place_option_order` | Single-leg put **or** multi-leg collar |
| `get_all_positions` | Current book for the quant engine |
| `get_account_info` | Buying power for the policy filter |
| `get_portfolio_history` | Equity curve for drawdown metrics |
| `get_stock_bars` | Returns series for the covariance matrix |
| `get_news` | Headline feed for the Sentiment Agent |

Constraint confirmed by Alpaca's team: **MCP option orders are market and limit only** — no stop orders. Use this in the write-up as a design justification, not a limitation.
