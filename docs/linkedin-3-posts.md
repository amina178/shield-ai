# Shield-AI — three LinkedIn posts

Written in the compressed register that earned the organiser's comment on other
participants' posts: short lines, concrete numbers, no marketing voice.

**Tag `@lablab.ai` and `@Alpaca` in the BODY, not in a comment.** That is what puts the
post in front of their social team — reach had nothing to do with it (the post that got
their attention had five reactions).

Post one per day. Four or five hashtags, not nine.

---

## POST 1 — today. The result.

> The window closed. My agent's stock book lost money.
>
> The account finished up.
>
> Shield-AI ran the official five-day window on a fresh $100,000 Alpaca paper account.
> Final equity $100,126.15.
>
> Here is the decomposition, and it is the whole thesis in three lines:
>
> Equity book, 6 inverse-volatility weighted positions — **minus $80.92**
> Protective puts, 15 contracts on NVDA — **plus $207.07**
> Net — **plus $126.15**
>
> The hedge did not just cover the loss. It was the entire P&L.
>
> And it did not pay off because NVDA fell. NVDA barely moved — 24 shares closed 72 cents
> below cost. The puts gained because implied volatility rose toward what my GARCH model had
> already forecast. The agent bought protection when options were pricing 8 volatility points
> below its own forecast of realised vol, and the market repriced.
>
> One trade is not an edge. I am not going to pretend otherwise. But the mechanism did exactly
> what it was designed to do, on a live account, with the order ID in an append-only log.
>
> Risk management — not alpha — is the killer app for agentic finance.
>
> Repo below. Paper. Public.
>
> @lablab.ai @Alpaca
> #lablabai #Fintech #RiskManagement #AIAgents

---

## POST 2 — tomorrow. The units bug.

> My VaR model said the portfolio risk was 2.57%.
>
> I halved every position. It still said 2.57%.
>
> Halving a book has to halve its risk. That is not a modelling opinion, it is arithmetic.
> So something was broken — and every individual formula checked out.
>
> The bug lived at the seam. My estimators take normalised weights, so they return risk as a
> fraction of **invested notional**. My mandate was written as a fraction of **account equity**.
> At 55% invested those differ by almost a factor of two.
>
> Worse: scaling every position down proportionally leaves normalised weights unchanged. So the
> position-sizing loop could shrink the book toward zero while the reported risk never moved.
> It would have cut exposure forever and never come into compliance.
>
> Two different meanings of the word "fraction," silently assumed to be the same.
>
> That class of bug is not caught by checking formulas. Every formula was right. It is caught by
> checking invariants: halve the book, halve the risk. That is now a regression test.
>
> Building an autonomous agent for the @lablab.ai × @Alpaca hackathon taught me that the
> dangerous errors are not in the mathematics. They are in the assumptions between the pieces.
>
> #lablabai #QuantitativeFinance #RiskManagement #BuildInPublic

---

## POST 3 — day after. The discipline post.

> The most attractive signal my agent found all week, it refused to trade.
>
> MSFT. GARCH forecasting 38.8% realised volatility, options implying 24.1%. Protection was
> 14.6 volatility points cheap — the widest gap of the entire run.
>
> Blocked. Open interest 429, below my liquidity floor of 1,000.
>
> I set that threshold before the scoring window opened. When it started refusing the trades I
> wanted, I left it alone.
>
> Here is why. An option you cannot exit in a crisis is not a hedge. The price you would pay to
> unwind it is unbounded precisely on the day you need to. A rich signal does not change that —
> and loosening a risk parameter after watching it block something you wanted is not tuning. It
> is the failure mode the whole system exists to prevent.
>
> The log records the refusal, the reason, and the number. Four other names were refused the same
> cycle.
>
> A record of executed trades proves an agent traded. A record of refusals proves it had
> judgement.
>
> That is the part I would want a risk committee to read.
>
> @lablab.ai @Alpaca
> #lablabai #RiskManagement #AIAgents #Fintech

---

## Notes

**Post 1 is the strongest — lead with it.** "My stock book lost money and the account finished
up" is a hook that stops a scroll, and every number behind it is verifiable.

**Reply to every comment**, especially an organiser's. The participant whose post got noticed
replied within the hour and kept the thread alive, which sent a second notification to the page.

**Do not post more than one per day.** A second post the same day cannibalises the first.

**Screenshots worth attaching** — first comment, not the body:
- Post 1: the terminal output showing the NVDA hedge with its order ID
- Post 2: the failing test output, or the diff of the fix
- Post 3: the blocked-by-liquidity line from the agent log
