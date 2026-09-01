"""
Shield-AI — deterministic policy filter.

This is the last gate before the market, and nothing the language model says can
open it. The LLM proposes a hedge; this module decides whether that proposal is
allowed to exist. Every check returns a named reason on failure, and that reason
goes straight into the audit log — so a refusal is documented, not silent.

Order matters. The cheapest and most disqualifying checks run first, so a
contract that fails on liquidity never consumes a buying-power calculation.
"""

from __future__ import annotations

from dataclasses import dataclass

from shield_ai.config import RiskMandate


@dataclass(frozen=True)
class PolicyVerdict:
    """Allowed or not, and if not, exactly which gate said no."""

    allowed: bool
    gate: str = ""
    reason: str = ""

    @classmethod
    def ok(cls) -> "PolicyVerdict":
        return cls(allowed=True)

    @classmethod
    def block(cls, gate: str, reason: str) -> "PolicyVerdict":
        return cls(allowed=False, gate=gate, reason=reason)


# Liquidity thresholds. An illiquid option is not a hedge: you can buy it, but
# the price you would pay to get out of it in a crisis is unbounded, and the
# quoted mid is fiction.
#
# The thresholds depend on WHERE the quote came from, and this distinction is
# not a loosening of standards — it is measuring the right thing.
#
#   opra       — real consolidated exchange quotes. A 5% spread on a liquid
#                contract is genuinely wide, so the spread test is meaningful
#                and strict.
#   indicative — Alpaca's free synthetic feed. It does not publish real
#                two-sided markets, so its spreads are wide by construction:
#                SPY options, the most liquid contracts in existence and
#                normally quoted a penny wide, come back at 25%. Applying the
#                OPRA threshold here does not measure liquidity, it measures
#                the feed. Open interest — which is real data either way —
#                becomes the primary test, and the spread check is kept only
#                as a sanity bound against a genuinely broken quote.
MAX_SPREAD_PCT = 0.05
MAX_SPREAD_PCT_INDICATIVE = 0.60
MIN_OPEN_INTEREST = 500
MIN_OPEN_INTEREST_INDICATIVE = 1_000   # stricter, since spread tells us little
MIN_ABSOLUTE_BID = 0.05      # below this, the contract is effectively worthless


def spread_pct(bid: float | None, ask: float | None) -> float | None:
    """Relative bid/ask spread. None when the quote is unusable."""
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
        return None
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid if mid > 0 else None


def midpoint(bid: float | None, ask: float | None) -> float | None:
    """
    Natural midpoint, used as the limit price.

    Pricing at the mid instead of crossing the spread is free money on every
    trade: on a $2.00/$2.20 quote, paying the ask costs 10 cents a share, which
    is $10 a contract. Over a week of premium selling that is a meaningful part
    of the edge — and MCP supports limit orders, so there is no reason to give
    it away with a market order.
    """
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return None
    return round((bid + ask) / 2.0, 2)


def check_liquidity(
    contract: dict,
    snapshot: dict,
    quote_source: str = "opra",
) -> PolicyVerdict:
    """
    Reject anything we could not exit in a hurry.

    `quote_source` selects the threshold set — see the constants above for why
    the indicative feed needs different ones. The source is recorded in the
    refusal message so the audit log always shows which standard was applied.
    """
    indicative = quote_source.lower() == "indicative"
    max_spread = MAX_SPREAD_PCT_INDICATIVE if indicative else MAX_SPREAD_PCT
    min_oi = MIN_OPEN_INTEREST_INDICATIVE if indicative else MIN_OPEN_INTEREST

    bid, ask = snapshot.get("bid"), snapshot.get("ask")

    if bid is None or ask is None:
        return PolicyVerdict.block("liquidity", "no two-sided quote available")

    if bid < MIN_ABSOLUTE_BID:
        return PolicyVerdict.block(
            "liquidity", f"bid {bid:.2f} below minimum {MIN_ABSOLUTE_BID:.2f}"
        )

    # Open interest first on the indicative feed: it is real exchange data
    # regardless of which quote feed you subscribe to, so it is the more
    # trustworthy liquidity signal of the two.
    oi = contract.get("open_interest", 0)
    if oi < min_oi:
        return PolicyVerdict.block(
            "liquidity",
            f"open interest {oi} below {min_oi} ({quote_source} feed)",
        )

    sp = spread_pct(bid, ask)
    if sp is None:
        return PolicyVerdict.block("liquidity", f"unusable quote {bid}/{ask}")
    if sp > max_spread:
        return PolicyVerdict.block(
            "liquidity",
            f"spread {sp:.1%} exceeds {max_spread:.0%} ({quote_source} feed)",
        )

    return PolicyVerdict.ok()


def check_hedge_cost(
    premium_per_contract: float,
    contracts: int,
    equity: float,
    mandate: RiskMandate,
) -> PolicyVerdict:
    """
    Cap what protection may cost.

    An unbounded hedge budget converts a risk system into a slow bleed: you can
    always buy more insurance, and eventually the insurance is the loss.
    """
    # Option premiums are quoted per share; a contract covers 100 shares.
    total = premium_per_contract * 100.0 * contracts
    cap = equity * mandate.max_hedge_cost_pct

    if total > cap:
        return PolicyVerdict.block(
            "hedge_cost",
            f"${total:,.0f} exceeds the {mandate.max_hedge_cost_pct:.1%} "
            f"cap of ${cap:,.0f}",
        )
    return PolicyVerdict.ok()


def check_buying_power(
    required: float,
    options_buying_power: float,
) -> PolicyVerdict:
    """Never submit an order the account cannot fund."""
    if required > options_buying_power:
        return PolicyVerdict.block(
            "buying_power",
            f"needs ${required:,.0f}, account has ${options_buying_power:,.0f}",
        )
    return PolicyVerdict.ok()


def check_coverage(
    side: str,
    contract_type: str,
    contracts: int,
    shares_held: int,
    cash_available: float,
    strike: float,
    mandate: RiskMandate,
) -> PolicyVerdict:
    """
    Enforce that every short option is covered.

    A short call without the underlying has theoretically unlimited loss, which
    is categorically incompatible with a risk-guardrails agent — no confidence
    level, no LLM rationale and no attractive premium can make it acceptable.
    A short put must be cash-secured for the same reason in the other direction.
    """
    if side.upper() != "SELL":
        return PolicyVerdict.ok()

    if contract_type.lower() == "call":
        needed = contracts * 100
        if shares_held < needed:
            if not mandate.allow_naked_short_calls:
                return PolicyVerdict.block(
                    "coverage",
                    f"short {contracts} call(s) needs {needed} shares, "
                    f"holding {shares_held} — naked calls are forbidden",
                )
        return PolicyVerdict.ok()

    if contract_type.lower() == "put":
        needed_cash = contracts * 100 * strike
        if cash_available < needed_cash:
            return PolicyVerdict.block(
                "coverage",
                f"cash-secured put needs ${needed_cash:,.0f}, "
                f"have ${cash_available:,.0f}",
            )
        return PolicyVerdict.ok()

    return PolicyVerdict.block("coverage", f"unknown contract type {contract_type!r}")


def check_options_level(
    strategy: str,
    options_trading_level: int,
) -> PolicyVerdict:
    """
    Fail fast on account permissions rather than at the broker.

    Level 1 = covered calls and cash-secured puts. Level 2 adds long options.
    Level 3 adds multi-leg. Checking here turns a confusing broker rejection
    into a clear, logged reason.
    """
    required = {
        "covered_call": 1,
        "cash_secured_put": 1,
        "long_put": 2,
        "protective_put": 2,
        "collar": 3,
        "multileg": 3,
    }
    need = required.get(strategy, 3)
    if options_trading_level < need:
        return PolicyVerdict.block(
            "options_level",
            f"{strategy} needs level {need}, account has "
            f"{options_trading_level}",
        )
    return PolicyVerdict.ok()


def evaluate(
    *,
    strategy: str,
    side: str,
    contract: dict,
    snapshot: dict,
    contracts: int,
    equity: float,
    cash_available: float,
    options_buying_power: float,
    shares_held: int,
    options_trading_level: int,
    mandate: RiskMandate,
    sentiment_veto: str = "",
    quote_source: str = "opra",
) -> PolicyVerdict:
    """
    Run every gate in order and return the first refusal.

    The sentiment veto is checked FIRST, before any pricing work. If the news
    layer has flagged event risk on this underlying, no amount of attractive
    premium makes the trade acceptable — that is the entire reason the cognitive
    layer exists, and putting it last would let a cheap contract sneak past on
    a technicality.
    """
    if sentiment_veto:
        return PolicyVerdict.block("sentiment_veto", sentiment_veto)

    for verdict in (
        check_options_level(strategy, options_trading_level),
        check_liquidity(contract, snapshot, quote_source),
        check_coverage(
            side, contract.get("type", ""), contracts, shares_held,
            cash_available, float(contract.get("strike", 0.0)), mandate,
        ),
    ):
        if not verdict.allowed:
            return verdict

    mid = midpoint(snapshot.get("bid"), snapshot.get("ask"))
    if mid is None:
        return PolicyVerdict.block("pricing", "cannot compute a midpoint")

    if side.upper() == "BUY":
        cost = check_hedge_cost(mid, contracts, equity, mandate)
        if not cost.allowed:
            return cost
        bp = check_buying_power(mid * 100 * contracts, options_buying_power)
        if not bp.allowed:
            return bp

    return PolicyVerdict.ok()
