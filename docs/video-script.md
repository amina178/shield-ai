# Shield-AI — video presentation script

**Target: 2 minutes 30 seconds.** Most hackathon judges watch the first 30 seconds
of every video and the rest of only a few. The hook is the whole game.

**Recording setup:** QuickTime → File → New Screen Recording (macOS built-in). Record
your screen with the terminal and browser, and narrate live. No editing needed if you
rehearse the run once. Keep the terminal font large — judges often watch on a laptop.

**Before you hit record**, have these open and ready:
- Terminal in the project folder, agent NOT yet running
- Browser tab 1: the Alpaca paper dashboard showing your positions
- Browser tab 2: `logs/decisions.jsonl` or the GitHub repo
- The volatility smile image open in Preview

---

## 0:00–0:25 · The hook

> *(Slide 1 or just your face / the cover image on screen)*

"Most AI trading agents are one bad headline away from blowing up an account.

Hand a language model a brokerage key, tell it to pick winners, and you haven't built
a quant system — you've built a leveraged guess. No covariance matrix, no tail
estimate, no loss budget.

I'm Amina, a risk data analyst, and I built the other half of the stack. This is
Shield-AI. It doesn't predict the market. It defends the downside."

---

## 0:25–0:50 · Why this architecture

> *(Slide 3 — the scoring rule)*

"The design came from the scoring rule. Performance is judged on terminal account
equity, with no benchmark. In a five-day window, a thirty percent drawdown is
unrecoverable.

So maximising final equity isn't a prediction problem. It's a risk-management
problem. That single observation determined everything that follows."

---

## 0:50–1:20 · How it works

> *(Slide 4 — architecture, then switch to the terminal)*

"A Python quant engine computes 99% Portfolio Value-at-Risk three independent ways,
Conditional VaR, and beta-weighted Delta. A GARCH model forecasts realised
volatility.

Then the same GARCH model does a second job: I compare its forecast to what options
are implying. That gap is the variance risk premium — it's the trading signal. One
model, two jobs.

The language model stays asleep until a guardrail breaks."

---

## 1:20–1:50 · Live run

> **Run the agent on camera.** `PYTHONPATH=. python3 -u scripts/run_agent.py`

"Here it is running against my live Alpaca paper account.

It reads the book, computes the risk, forecasts volatility per name, pulls the option
chains, and decides.

*(point at the NVDA line)* Here — GARCH forecasts 41.9% realised volatility, options
are implying only 29.8%. Protection is twelve points cheap. It buys four protective
puts, three hundred and ninety-six dollars against a fifteen-hundred-dollar hedge
budget, and logs the broker order ID.

*(point at the blocked lines)* And here it refuses four other trades. Edge inside the
no-trade band, or open interest too thin to trust the quote. Every refusal is logged
with its reason."

---

## 1:50–2:15 · The bug — this is what makes you memorable

> *(Show the volatility smile image)*

"On the first live run it reported a 15.6 volatility-point edge on SPY. A spectacular
edge on the most efficiently priced instrument on earth is not an edge. It's a bug.

It was mine. I was taking the median implied volatility across the entire option
chain — but far out-of-the-money puts trade at much higher IV, because they're crash
insurance. I was measuring skew and calling it alpha.

Fixed it to read from a single delta-matched contract. The edge fell from 15.6 to
4.7, and there's now a regression test guarding it."

---

## 2:15–2:30 · Close

> *(Slide 8 or 9)*

"Forty-seven verification tests, including Kupiec and Christoffersen backtests that
validate the risk model statistically rather than asserting it works.

A record of only executed trades proves an agent traded. A record of refusals proves
it had judgement.

Risk management — not alpha — is the killer app for agentic finance. Thank you."

---

## Delivery notes

**Speak slower than feels natural.** Nervous recording runs about 20% fast, and this
script is dense. If you finish under 2:15 you were rushing — record again.

**Do not apologise for anything on screen.** If a trade doesn't fire during the live
run, that IS the demo: the agent found nothing worth trading and said so. Narrate it
as the system working, because it is.

**One take is fine.** Judges reward a clear explanation over production value. A
polished video of a vague idea loses to a plain screen recording of a sharp one.

**If you record only 60 seconds**, keep the hook, the NVDA trade, and the bug. Drop
everything else. Those three beats carry the whole project.
