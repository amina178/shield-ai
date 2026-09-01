"""
Shield-AI — verification of the cognitive layer.

The decision tree encodes the project's entire thesis, so each branch is tested
explicitly. In particular: a VaR breach must outrank an attractive premium. If
that test ever fails, Shield-AI has quietly become an income strategy wearing a
risk-management costume.
"""

from __future__ import annotations

from shield_ai.config import MANDATE
from shield_ai.agents.sentiment import assess, scan_headlines
from shield_ai.agents.risk_manager import (
    decide_action, select_contract, size_by_risk_budget, build_proposal,
)


def _news(headline: str, symbols: list[str], summary: str = "") -> dict:
    return {"headline": headline, "summary": summary, "symbols": symbols,
            "created_at": "2026-08-31T12:00:00Z", "source": "test", "url": ""}


# ---------------------------------------------------------------------------
# Sentiment agent
# ---------------------------------------------------------------------------

def test_sentiment_flags_earnings():
    news = [_news("Apple to report Q3 earnings on Thursday", ["AAPL"])]
    v = assess("AAPL", news, use_llm=False)
    assert v.has_event_risk and v.severity == 3
    assert "earnings" in v.categories
    print(f"  severity {v.severity}, categories {v.categories}")
    print(f"  veto: {v.veto_reason}  ✓")


def test_sentiment_flags_regulatory():
    news = [_news("FDA decision on lead candidate expected next week", ["XYZ"])]
    v = assess("XYZ", news, use_llm=False)
    assert v.blocks_premium_selling and "regulatory" in v.categories
    print(f"  FDA headline -> severity {v.severity}, blocks selling  ✓")


def test_sentiment_ignores_routine_coverage():
    news = [
        _news("Microsoft opens new campus in Dublin", ["MSFT"]),
        _news("Analysts discuss cloud market share trends", ["MSFT"]),
    ]
    v = assess("MSFT", news, use_llm=False)
    assert not v.has_event_risk and v.severity == 0
    print(f"  {v.headlines_reviewed} routine headlines -> no veto  ✓")


def test_sentiment_ignores_other_symbols():
    """A catalyst on NVDA must not veto a trade on SPY."""
    news = [_news("NVIDIA reports earnings Wednesday", ["NVDA"])]
    v = assess("SPY", news, use_llm=False)
    assert not v.has_event_risk
    print("  NVDA earnings does not veto SPY  ✓")


def test_sentiment_captures_evidence():
    news = [_news("Board announces merger with rival", ["ACME"])]
    v = scan_headlines("ACME", news)
    assert v.evidence and "merger" in v.evidence[0].lower()
    print(f"  evidence recorded: {v.evidence[0][:60]}  ✓")


# ---------------------------------------------------------------------------
# Decision tree — the thesis
# ---------------------------------------------------------------------------

def test_var_breach_outranks_rich_premium():
    """
    The single most important behaviour in the system. A very attractive edge
    (+12 vol points) must NOT produce a premium sale when VaR is over the limit.
    """
    action, why = decide_action(
        var_headline=0.031,          # over the 2.0% mandate
        net_delta=0.60,
        edge_points=+12.0,           # extremely rich premium
        mandate=MANDATE,
    )
    assert action == "HEDGE", f"got {action}: {why}"
    print(f"  VaR 3.1% + edge +12.0 pts -> {action}")
    print(f"  {why}  ✓")


def test_rich_premium_sells_when_risk_is_quiet():
    action, why = decide_action(0.010, 0.50, +6.0, MANDATE)
    assert action == "SELL_PREMIUM"
    print(f"  VaR 1.0% + edge +6.0 -> {action}: {why}  ✓")


def test_cheap_options_buy_protection():
    action, why = decide_action(0.010, 0.50, -7.0, MANDATE)
    assert action == "BUY_PROTECTION"
    print(f"  edge -7.0 -> {action}: {why}  ✓")


def test_warn_zone_blocks_new_premium():
    """Between warn and limit, the agent stops adding risk without yet hedging."""
    action, why = decide_action(0.018, 0.50, +8.0, MANDATE)
    assert action == "HOLD"
    print(f"  VaR 1.8% (warn zone) + edge +8.0 -> {action}")
    print(f"  {why}  ✓")


def test_small_edge_does_nothing():
    action, why = decide_action(0.010, 0.50, +2.0, MANDATE)
    assert action == "HOLD"
    print(f"  edge +2.0 inside no-trade band -> {action}  ✓")


def test_delta_breach_hedges():
    action, why = decide_action(0.010, 0.95, +6.0, MANDATE)
    assert action == "HEDGE"
    print(f"  net delta 0.95 above band -> {action}: {why}  ✓")


# ---------------------------------------------------------------------------
# Contract selection and sizing
# ---------------------------------------------------------------------------

CONTRACTS = [
    {"symbol": "SPY_P590", "strike": 590.0, "type": "put", "dte": 14, "open_interest": 3000},
    {"symbol": "SPY_P600", "strike": 600.0, "type": "put", "dte": 14, "open_interest": 5000},
    {"symbol": "SPY_P610", "strike": 610.0, "type": "put", "dte": 14, "open_interest": 4000},
]
SNAPS = {
    "SPY_P590": {"delta": -0.10, "bid": 1.00, "ask": 1.04, "implied_volatility": 0.20},
    "SPY_P600": {"delta": -0.21, "bid": 2.00, "ask": 2.06, "implied_volatility": 0.22},
    "SPY_P610": {"delta": -0.35, "bid": 3.50, "ask": 3.60, "implied_volatility": 0.24},
}


def test_select_contract_targets_delta():
    contract, snap = select_contract(CONTRACTS, SNAPS, target_delta=0.20)
    assert contract["symbol"] == "SPY_P600", contract["symbol"]
    print(f"  target delta 0.20 -> picked {contract['symbol']} "
          f"(delta {snap['delta']:+.2f})  ✓")


def test_select_contract_skips_unquotable():
    snaps = dict(SNAPS)
    snaps["SPY_P600"] = {"delta": -0.20, "bid": None, "ask": None}
    contract, _ = select_contract(CONTRACTS, snaps, target_delta=0.20)
    assert contract["symbol"] != "SPY_P600"
    print(f"  unquotable contract skipped -> picked {contract['symbol']}  ✓")


def test_sizing_shrinks_as_var_rises():
    """
    Size must fall monotonically as risk rises, and reach zero at the limit.

    Note the steps are not smooth: contracts are integers and one costs $203, so
    once the remaining budget drops below that the answer jumps straight to 0.
    That is correct — a fractional option does not exist — and it is why this
    asserts a non-increasing sequence rather than a strictly decreasing one.
    """
    price = 2.03
    sizes = [
        (var, size_by_risk_budget(price, 100_000.0, var, MANDATE))
        for var in (0.002, 0.005, 0.012, 0.018, 0.025)
    ]

    values = [n for _, n in sizes]
    assert values == sorted(values, reverse=True), values
    assert values[0] > values[-1], "size must fall as risk rises"
    assert values[-1] == 0, "a breached book must be allowed zero new contracts"

    for var, n in sizes:
        headroom = max(MANDATE.var_limit_pct - var, 0.0) * 100_000.0
        print(f"  VaR {var:.1%}  headroom ${headroom:>6,.0f}  -> {n} contract(s)")
    print("  monotonically non-increasing, zero at the limit  ✓")


def test_sizing_never_exceeds_half_the_headroom():
    """One trade must never be able to push the book from compliant to breached."""
    equity, var = 100_000.0, 0.010
    price = 2.00
    n = size_by_risk_budget(price, equity, var, MANDATE)
    headroom_dollars = (MANDATE.var_limit_pct - var) * equity
    assert n * price * 100 <= headroom_dollars * 0.5 + 1e-6
    print(f"  headroom ${headroom_dollars:,.0f}, position "
          f"${n * price * 100:,.0f} <= half  ✓")


def test_build_proposal_end_to_end():
    p = build_proposal(
        action="SELL_PREMIUM", rationale="IV 6.0 pts above forecast",
        symbol="SPY", contracts=CONTRACTS, snapshots=SNAPS,
        equity=100_000.0, var_headline=0.010, mandate=MANDATE,
    )
    assert p.action == "SELL_PREMIUM"
    assert p.side == "SELL" and p.contracts > 0
    assert p.limit_price == 2.03
    print(f"  {p.strategy} x{p.contracts} at {p.limit_price}")
    print(f"  {p.rationale[:100]}...  ✓")


def test_build_proposal_holds_when_budget_is_zero():
    p = build_proposal(
        action="SELL_PREMIUM", rationale="rich premium",
        symbol="SPY", contracts=CONTRACTS, snapshots=SNAPS,
        equity=100_000.0, var_headline=0.0199, mandate=MANDATE,
    )
    assert p.action == "HOLD" and "0 contracts" in p.rationale
    print(f"  no headroom -> {p.action}: {p.rationale[-40:]}  ✓")


def test_selection_refuses_a_far_off_delta():
    """
    Regression: the SPY case from the first live run.

    Only far out-of-the-money contracts were priced, so "closest available"
    returned a 3-delta option against a 20-delta target. Its IV sits far out on
    the volatility smile, and comparing that to a GARCH forecast manufactures an
    edge that does not exist. The selector must return nothing instead.
    """
    far_only = [{"symbol": "SPY_P600", "strike": 600.0, "type": "put",
                 "dte": 7, "open_interest": 90}]
    snaps = {"SPY_P600": {"delta": -0.03, "bid": 0.10, "ask": 0.12,
                          "implied_volatility": 0.163}}

    assert select_contract(far_only, snaps, target_delta=0.20) is None
    # But it is accepted when the target itself is that far out.
    assert select_contract(far_only, snaps, target_delta=0.05) is not None
    print("  3-delta contract vs 20-delta target -> refused")
    print("  same contract vs 5-delta target     -> accepted  \u2713")


# ---------------------------------------------------------------------------

def main() -> int:
    tests = [
        ("Sentiment flags earnings", test_sentiment_flags_earnings),
        ("Sentiment flags regulatory events", test_sentiment_flags_regulatory),
        ("Sentiment ignores routine coverage", test_sentiment_ignores_routine_coverage),
        ("Sentiment scoped to the right symbol", test_sentiment_ignores_other_symbols),
        ("Sentiment records evidence", test_sentiment_captures_evidence),
        ("VaR breach outranks rich premium", test_var_breach_outranks_rich_premium),
        ("Rich premium sells when quiet", test_rich_premium_sells_when_risk_is_quiet),
        ("Cheap options buy protection", test_cheap_options_buy_protection),
        ("Warn zone blocks new premium", test_warn_zone_blocks_new_premium),
        ("Small edge does nothing", test_small_edge_does_nothing),
        ("Delta breach hedges", test_delta_breach_hedges),
        ("Contract selection targets delta", test_select_contract_targets_delta),
        ("Selection skips unquotable contracts", test_select_contract_skips_unquotable),
        ("Sizing shrinks as VaR rises", test_sizing_shrinks_as_var_rises),
        ("Sizing caps at half the headroom", test_sizing_never_exceeds_half_the_headroom),
        ("Proposal built end to end", test_build_proposal_end_to_end),
        ("Proposal holds with no budget", test_build_proposal_holds_when_budget_is_zero),
        ("Selection refuses a far-off delta (regression)", test_selection_refuses_a_far_off_delta),
    ]

    failures = 0
    for i, (name, fn) in enumerate(tests, 1):
        print(f"\n[{i:2d}] {name}")
        try:
            fn()
        except AssertionError as exc:
            print(f"  ✗ FAILED: {exc}")
            failures += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ ERROR: {type(exc).__name__}: {exc}")
            failures += 1

    print("\n" + "=" * 62)
    print(f"{len(tests) - failures}/{len(tests)} passed")
    print("=" * 62)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
