# Shield-AI — LinkedIn post (live agent)

*~290 words. Real numbers from the 1 September cycle.*

---

My AI trading agent found a 15.6 volatility-point edge on SPY.

It was my bug.

I'm building 𝗦𝗵𝗶𝗲𝗹𝗱-𝗔𝗜 for the Lablab.ai × Alpaca AI Trading Agents Hackathon — an autonomous risk agent that doesn't predict direction, it defends the downside. It sells option premium when implied volatility is rich against a GARCH forecast, buys protection when it's cheap, and hedges whenever Value-at-Risk breaches its mandate.

On the first live run it reported a spectacular edge on SPY. Spectacular edges on the most efficiently priced instrument on earth are not edges. They are bugs.

They were. I was taking the **median implied volatility across the entire option chain** and comparing it to my volatility forecast. But options have a smile: far out-of-the-money puts trade at much higher IV because they're crash insurance. My median sat way out on that wing. I was measuring skew and calling it alpha.

Fixed it to read IV from a single delta-matched contract. The SPY edge fell from +15.6 to +4.7 points.

Then the agent placed its first real trade — and it wasn't the one I expected:

📉 NVDA · GARCH forecast 41.9% realised vol · options implying only 29.8%
🛡️ Protection is 12 points cheap → bought 4 protective puts, $396 against a $1,500 hedge budget
📋 Order ID logged, VaR recomputed, decision written to an append-only audit trail

Four other names: no trade. Edge inside the no-trade band, or open interest too thin to trust the quote. Every refusal logged with its reason.

That log is the deliverable. A record of only executed trades proves an agent traded. A record of refusals proves it had judgement.

46 tests, including Kupiec and Christoffersen backtests that validate the risk model statistically rather than asserting it works.

Risk management — not alpha — is the killer app for agentic finance.

#Fintech #AIAgents #RiskManagement #QuantitativeFinance #AlpacaMarkets #OptionsTrading #BuildInPublic #lablabai

---

## Why this post works

The hook is a confession, and confessions outperform announcements on LinkedIn by a wide margin. Everyone else in the hackathon is posting "my agent found an edge." You're posting "my agent found an edge and I proved it was fake" — which is what an actual risk analyst does, and it signals seniority far more strongly than a win claim.

The NVDA trade is real evidence with an order ID behind it. The four refusals are the thesis in action.

## Optional second post (Day 4)

If you want a follow-up, the strongest remaining story is the units bug:

> My VaR model said the risk was 2.57%. I halved every position. It still said 2.57%.
>
> Every formula was correct. The bug lived at the seam: the estimators returned risk as a fraction of *invested notional*, the mandate was written as a fraction of *account equity*. Two different meanings of "fraction," silently assumed to be the same.
>
> That class of bug isn't caught by checking formulas. It's caught by checking invariants: halve the book, halve the risk.
