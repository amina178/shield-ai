"""
Shield-AI — verification of the evidence and policy layers.

The policy filter is the component where a bug is most expensive: a gate that
silently passes is how a risk agent ends up short a naked call. So each gate is
tested in both directions — it must ACCEPT what is legitimate and REJECT what
is not. A gate only tested on the happy path is not tested.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from shield_ai.config import MANDATE
from shield_ai.execution.audit import (
    AuditLog, DecisionRecord, TradeRecord, build_tearsheet, _utc_now,
)
from shield_ai.execution import policy


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def test_audit_log_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        log = AuditLog(tmp)
        cycle = log.new_cycle_id()

        log.log_decision(DecisionRecord(
            timestamp=_utc_now(), cycle_id=cycle, symbol="SPY",
            decision="SELL_PREMIUM", outcome="EXECUTED",
            var_headline=0.014, var_limit=0.020, edge_points=6.2,
            reason="IV 6.2 pts above GARCH forecast",
        ))
        log.log_decision(DecisionRecord(
            timestamp=_utc_now(), cycle_id=cycle, symbol="NVDA",
            decision="SELL_PREMIUM", outcome="VETOED",
            edge_points=9.1, rejected_by="sentiment_veto",
            reason="earnings in 2 days",
        ))
        log.log_trade(TradeRecord(
            timestamp=_utc_now(), cycle_id=cycle,
            broker_order_id="abc-123-def", symbol="SPY260918P00600000",
            side="SELL", qty=1, order_type="limit", limit_price=2.15,
            asset_class="option", strategy="cash_secured_put", status="accepted",
        ))

        d, t = log.decisions(), log.trades()
        assert len(d) == 2 and len(t) == 1
        # The refusal must survive the round trip — it is the evidence.
        vetoed = d[d["outcome"] == "VETOED"].iloc[0]
        assert vetoed["rejected_by"] == "sentiment_veto"
        assert t.iloc[0]["broker_order_id"] == "abc-123-def"
        print(f"  {len(d)} decisions, {len(t)} trades round-tripped")
        print(f"  refusal preserved: {vetoed['symbol']} <- {vetoed['reason']}  ✓")


def test_audit_log_survives_truncated_line():
    """An interrupted write must cost one record, never the whole file."""
    with tempfile.TemporaryDirectory() as tmp:
        log = AuditLog(tmp)
        log.log_decision(DecisionRecord(
            timestamp=_utc_now(), cycle_id="c1", symbol="SPY",
            decision="HOLD", outcome="NO_ACTION", reason="within mandate",
        ))
        # Simulate a crash mid-write.
        with open(log.decisions_path, "a") as fh:
            fh.write('{"timestamp":"2026-09-01T10:00:00","cyc')

        df = log.decisions()
        assert len(df) == 1, "the complete record must survive"
        print(f"  1 good record kept, malformed tail skipped  ✓")


# ---------------------------------------------------------------------------
# Liquidity gate
# ---------------------------------------------------------------------------

GOOD_CONTRACT = {"symbol": "SPY260918P00600000", "strike": 600.0,
                 "type": "put", "open_interest": 4200, "dte": 14}
GOOD_SNAP = {"bid": 2.00, "ask": 2.06, "delta": -0.20, "implied_volatility": 0.22}


def test_liquidity_accepts_a_liquid_contract():
    v = policy.check_liquidity(GOOD_CONTRACT, GOOD_SNAP)
    assert v.allowed, v.reason
    sp = policy.spread_pct(2.00, 2.06)
    print(f"  spread {sp:.2%}, OI {GOOD_CONTRACT['open_interest']} -> allowed  ✓")


def test_liquidity_rejects_wide_spread():
    v = policy.check_liquidity(GOOD_CONTRACT, {"bid": 1.00, "ask": 1.40})
    assert not v.allowed and v.gate == "liquidity"
    print(f"  spread {policy.spread_pct(1.00,1.40):.1%} -> blocked: {v.reason}  ✓")


def test_liquidity_rejects_thin_open_interest():
    thin = dict(GOOD_CONTRACT, open_interest=12)
    v = policy.check_liquidity(thin, GOOD_SNAP)
    assert not v.allowed and "open interest" in v.reason
    print(f"  OI 12 -> blocked: {v.reason}  ✓")


def test_liquidity_rejects_missing_quote():
    v = policy.check_liquidity(GOOD_CONTRACT, {"bid": None, "ask": None})
    assert not v.allowed
    print(f"  no quote -> blocked: {v.reason}  ✓")


def test_midpoint_pricing():
    assert policy.midpoint(2.00, 2.06) == 2.03
    assert policy.midpoint(None, 2.06) is None
    print("  mid(2.00, 2.06) = 2.03  ✓")


# ---------------------------------------------------------------------------
# Coverage — the gate that must never leak
# ---------------------------------------------------------------------------

def test_naked_call_is_always_rejected():
    """The single most important refusal in the system."""
    v = policy.check_coverage(
        side="SELL", contract_type="call", contracts=2, shares_held=0,
        cash_available=1_000_000.0, strike=600.0, mandate=MANDATE,
    )
    assert not v.allowed and v.gate == "coverage"
    print(f"  sell 2 calls holding 0 shares -> blocked: {v.reason}  ✓")


def test_covered_call_is_allowed():
    v = policy.check_coverage(
        side="SELL", contract_type="call", contracts=2, shares_held=200,
        cash_available=0.0, strike=600.0, mandate=MANDATE,
    )
    assert v.allowed
    print("  sell 2 calls holding 200 shares -> allowed  ✓")


def test_cash_secured_put_requires_cash():
    blocked = policy.check_coverage(
        side="SELL", contract_type="put", contracts=1, shares_held=0,
        cash_available=5_000.0, strike=600.0, mandate=MANDATE,
    )
    allowed = policy.check_coverage(
        side="SELL", contract_type="put", contracts=1, shares_held=0,
        cash_available=61_000.0, strike=600.0, mandate=MANDATE,
    )
    assert not blocked.allowed and allowed.allowed
    print(f"  put needs $60,000 collateral: $5,000 blocked, $61,000 allowed  ✓")


# ---------------------------------------------------------------------------
# Cost, buying power, options level
# ---------------------------------------------------------------------------

def test_hedge_cost_cap():
    equity = 100_000.0
    cap = equity * MANDATE.max_hedge_cost_pct
    cheap = policy.check_hedge_cost(2.00, 5, equity, MANDATE)     # $1,000
    dear = policy.check_hedge_cost(12.00, 5, equity, MANDATE)     # $6,000
    assert cheap.allowed and not dear.allowed
    print(f"  cap ${cap:,.0f}: $1,000 allowed, $6,000 blocked  ✓")


def test_options_level_gate():
    assert policy.check_options_level("covered_call", 1).allowed
    assert not policy.check_options_level("protective_put", 1).allowed
    assert policy.check_options_level("protective_put", 2).allowed
    assert not policy.check_options_level("collar", 2).allowed
    assert policy.check_options_level("collar", 3).allowed
    print("  L1 covered call ok / L1 protective put blocked / "
          "L2 collar blocked / L3 collar ok  ✓")


# ---------------------------------------------------------------------------
# Full evaluation, including veto precedence
# ---------------------------------------------------------------------------

def test_sentiment_veto_wins_over_everything():
    """
    A perfect contract with a news veto must still be refused. If this test
    fails, the cognitive layer is decorative.
    """
    v = policy.evaluate(
        strategy="cash_secured_put", side="SELL",
        contract=GOOD_CONTRACT, snapshot=GOOD_SNAP, contracts=1,
        equity=100_000.0, cash_available=200_000.0,
        options_buying_power=200_000.0, shares_held=0,
        options_trading_level=3, mandate=MANDATE,
        sentiment_veto="FDA decision expected within the option's life",
    )
    assert not v.allowed and v.gate == "sentiment_veto"
    print(f"  liquid contract + veto -> blocked by {v.gate}  ✓")


def test_full_evaluation_passes_a_clean_trade():
    v = policy.evaluate(
        strategy="cash_secured_put", side="SELL",
        contract=GOOD_CONTRACT, snapshot=GOOD_SNAP, contracts=1,
        equity=100_000.0, cash_available=200_000.0,
        options_buying_power=200_000.0, shares_held=0,
        options_trading_level=3, mandate=MANDATE,
    )
    assert v.allowed, f"blocked by {v.gate}: {v.reason}"
    print("  clean cash-secured put -> allowed  ✓")


# ---------------------------------------------------------------------------
# Tearsheet
# ---------------------------------------------------------------------------

def test_tearsheet_on_known_series():
    """A steadily rising curve must show a positive Sharpe and a small drawdown."""
    rng = np.random.default_rng(3)
    n = 120
    rets = rng.normal(0.0006, 0.004, n)
    eq = pd.Series(100_000 * np.cumprod(1 + rets),
                   index=pd.date_range("2026-03-02", periods=n, freq="B"))
    pnl = pd.Series([120, -260, 95, 140, -80, 210, 60])

    ts = build_tearsheet(eq, pnl)
    assert ts.sharpe > 0
    assert ts.max_drawdown < 0
    assert ts.sortino >= ts.sharpe * 0.5
    print(f"  return {ts.total_return:+.2%}  Sharpe {ts.sharpe:.2f}  "
          f"Sortino {ts.sortino:.2f}  maxDD {ts.max_drawdown:.2%}")
    print(f"  win rate {ts.win_rate:.0%}  profit factor {ts.profit_factor:.2f}  "
          f"largest loss ${ts.largest_loss:,.0f}  ✓")


def test_tearsheet_detects_a_drawdown():
    """A curve that falls 20% and recovers must report a ~20% max drawdown."""
    eq = pd.Series(
        [100_000, 105_000, 84_000, 90_000, 103_000],
        index=pd.date_range("2026-03-02", periods=5, freq="B"),
    )
    ts = build_tearsheet(eq)
    assert abs(ts.max_drawdown - (84_000 / 105_000 - 1)) < 1e-9
    print(f"  peak 105,000 -> trough 84,000 gives "
          f"maxDD {ts.max_drawdown:.2%}  ✓")


# ---------------------------------------------------------------------------

def main() -> int:
    tests = [
        ("Audit log round-trip keeps refusals", test_audit_log_roundtrip),
        ("Audit log survives a truncated line", test_audit_log_survives_truncated_line),
        ("Liquidity accepts a liquid contract", test_liquidity_accepts_a_liquid_contract),
        ("Liquidity rejects a wide spread", test_liquidity_rejects_wide_spread),
        ("Liquidity rejects thin open interest", test_liquidity_rejects_thin_open_interest),
        ("Liquidity rejects a missing quote", test_liquidity_rejects_missing_quote),
        ("Midpoint pricing", test_midpoint_pricing),
        ("Naked call is always rejected", test_naked_call_is_always_rejected),
        ("Covered call is allowed", test_covered_call_is_allowed),
        ("Cash-secured put requires collateral", test_cash_secured_put_requires_cash),
        ("Hedge cost cap", test_hedge_cost_cap),
        ("Options level gate", test_options_level_gate),
        ("Sentiment veto outranks every gate", test_sentiment_veto_wins_over_everything),
        ("Full evaluation passes a clean trade", test_full_evaluation_passes_a_clean_trade),
        ("Tearsheet on a known series", test_tearsheet_on_known_series),
        ("Tearsheet detects a drawdown", test_tearsheet_detects_a_drawdown),
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
