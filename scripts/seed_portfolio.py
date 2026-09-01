"""
Shield-AI — seed the base portfolio.

This builds the equity book that the agent will then defend and harvest premium
against. It is a ONE-TIME SETUP SCRIPT run by a human, not an agent action —
which is why it uses alpaca-py directly rather than the MCP gateway. MCP is the
agent's execution path for hedges and premium trades; seeding the initial book
is not something the agent decides.

Safety: this script is DRY-RUN by default. It prints the plan and the resulting
risk profile, and places nothing. Add --execute to actually submit orders.

    PYTHONPATH=. python3 scripts/seed_portfolio.py             # plan only
    PYTHONPATH=. python3 scripts/seed_portfolio.py --execute   # place orders

Sizing method
-------------
Positions are inverse-volatility weighted, not equal-dollar. Equal dollar
amounts across a utility and a high-beta semiconductor name means the
semiconductor dominates the portfolio's risk while looking "balanced" in the
account view. Weighting by 1/sigma equalises each name's approximate risk
contribution, which is the whole point of a book you intend to run to a VaR
budget. The allocation is then checked against the mandate BEFORE any order is
sent — if the plan breaches VaR, it is scaled down rather than submitted.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from shield_ai.config import MANDATE
from shield_ai.data.alpaca_client import ShieldDataClient, AlpacaConfigError
from shield_ai.quant.risk import build_risk_report, log_returns

# Fraction of equity to deploy into stock. The remainder is kept as cash to
# collateralise cash-secured puts and to pay for protective puts without
# forcing a liquidation at the worst possible moment.
TARGET_INVESTED = 0.55

# Covered calls require 100 shares per contract. Where a name is affordable in
# round lots we round to 100 so the income leg can actually be written on it.
CONTRACT_LOT = 100


def inverse_vol_weights(closes: pd.DataFrame) -> pd.Series:
    """Weights proportional to 1/sigma, normalised to sum to 1."""
    rets = log_returns(closes)
    vol = rets.std()
    if (vol <= 0).any():
        raise ValueError(f"Zero volatility for {list(vol[vol <= 0].index)}")
    inv = 1.0 / vol
    return inv / inv.sum()


def build_plan(closes: pd.DataFrame, equity: float) -> pd.DataFrame:
    """Turn weights into a concrete share count per symbol."""
    weights = inverse_vol_weights(closes)
    last = closes.iloc[-1]
    budget = equity * TARGET_INVESTED

    rows = []
    for sym in closes.columns:
        target_dollars = budget * weights[sym]
        price = float(last[sym])
        raw_shares = target_dollars / price

        # Round to a 100-lot when that is affordable, so the name can carry a
        # covered call. Otherwise fall back to whole shares.
        if raw_shares >= CONTRACT_LOT:
            shares = int(round(raw_shares / CONTRACT_LOT) * CONTRACT_LOT)
        else:
            shares = int(raw_shares)

        rows.append({
            "symbol": sym,
            "price": price,
            "weight": float(weights[sym]),
            "shares": shares,
            "market_value": shares * price,
            "round_lots": shares // CONTRACT_LOT,
        })

    plan = pd.DataFrame(rows).set_index("symbol")
    return plan[plan["shares"] > 0]


def scale_to_mandate(
    plan: pd.DataFrame,
    closes: pd.DataFrame,
    equity: float,
    max_iter: int = 12,
) -> tuple[pd.DataFrame, object]:
    """
    Shrink the plan until its projected VaR fits the mandate.

    This is the core discipline of Shield-AI expressed at portfolio-construction
    time: size is never chosen and then justified, it is SOLVED FOR against the
    risk budget. Each iteration cuts exposure by 10% and re-prices the risk.
    """
    current = plan.copy()
    for i in range(max_iter):
        report = build_risk_report(
            closes,
            current["market_value"],
            equity=equity,
            confidence=MANDATE.confidence,
            student_t_df=MANDATE.student_t_df,
            mc_paths=20_000,
            benchmark=MANDATE.benchmark,
            min_observations=MANDATE.min_observations,
        )
        within_var = report.var_headline <= MANDATE.var_limit_pct
        within_delta = report.net_delta <= MANDATE.delta_band[1]

        if within_var and within_delta:
            if i:
                print(f"  scaled down {i} time(s) to fit the mandate")
            return current, report

        reason = "VaR" if not within_var else "net delta"
        print(f"  iteration {i + 1}: {reason} over limit "
              f"(VaR {report.var_headline:.2%} / delta {report.net_delta:.2f}) "
              f"-> cutting exposure 10%")

        current["shares"] = (current["shares"] * 0.9).astype(int)
        current = current[current["shares"] > 0]
        current["market_value"] = current["shares"] * current["price"]
        if current.empty:
            raise RuntimeError("Scaled to zero — the universe is too volatile "
                               "for this mandate. Loosen var_limit_pct or pick "
                               "lower-beta names.")

    raise RuntimeError("Could not fit the mandate within the iteration budget.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true",
                    help="actually submit orders (default is a dry run)")
    args = ap.parse_args()

    try:
        client = ShieldDataClient(paper=True)
    except AlpacaConfigError as exc:
        print(f"ERROR: {exc}")
        return 1

    acct = client.account()
    clock = client.market_clock()

    print("=" * 68)
    print("Shield-AI — portfolio seeding")
    print("=" * 68)
    print(f"equity          ${acct.equity:,.2f}")
    print(f"cash            ${acct.cash:,.2f}")
    print(f"options level   {acct.options_trading_level}")
    print(f"market          {'OPEN' if clock['is_open'] else 'CLOSED'}")
    print(f"mode            {'EXECUTE' if args.execute else 'DRY RUN'}")

    existing = client.positions()
    if not existing.empty:
        print(f"\nWARNING: {len(existing)} equity positions already exist:")
        for sym, val in existing.items():
            print(f"  {sym:6s} ${val:,.2f}")
        print("Seeding on top of an existing book will double up exposure.")
        if args.execute:
            print("Refusing to execute. Close positions first, or edit the universe.")
            return 1

    print(f"\nFetching {MANDATE.lookback_days}+ trading days for "
          f"{len(MANDATE.universe)} symbols...")
    closes = client.daily_closes(list(MANDATE.universe), lookback_days=400)
    print(f"  {closes.shape[0]} observations "
          f"({closes.index[0].date()} -> {closes.index[-1].date()})")

    print("\nBuilding inverse-volatility weighted plan...")
    plan = build_plan(closes, acct.equity)
    plan, report = scale_to_mandate(plan, closes, acct.equity)

    print("\n" + "-" * 68)
    print(f"{'symbol':<8}{'price':>10}{'weight':>9}{'shares':>9}"
          f"{'value':>13}{'lots':>7}")
    print("-" * 68)
    for sym, r in plan.iterrows():
        print(f"{sym:<8}{r['price']:>10,.2f}{r['weight']:>8.1%}"
              f"{int(r['shares']):>9}{r['market_value']:>13,.2f}"
              f"{int(r['round_lots']):>7}")
    print("-" * 68)
    invested = plan["market_value"].sum()
    print(f"{'TOTAL':<8}{'':>10}{'':>9}{'':>9}{invested:>13,.2f}")
    print(f"\ninvested        {invested / acct.equity:.1%} of equity")
    print(f"cash retained   ${acct.equity - invested:,.2f}")

    print("\nProjected risk profile:")
    print(f"  VaR headline  {report.var_headline:.3%} "
          f"(${report.var_dollars:,.0f})   limit {MANDATE.var_limit_pct:.2%}")
    print(f"  CVaR          {report.cvar:.3%}")
    print(f"  net delta     {report.net_delta:.3f}   "
          f"band {MANDATE.delta_band[0]:.2f}-{MANDATE.delta_band[1]:.2f}")
    print(f"  disagreement  {report.model_disagreement:.1%}")

    writable = int(plan["round_lots"].sum())
    print(f"\nCovered calls writable: {writable} contract(s) across "
          f"{int((plan['round_lots'] > 0).sum())} name(s)")
    if writable == 0:
        print("  WARNING: no round lots — the premium-selling leg has nothing "
              "to write against. Consider lower-priced names in the universe.")

    if not args.execute:
        print("\nDRY RUN — no orders placed. Re-run with --execute to submit.")
        return 0

    if not clock["is_open"]:
        print(f"\nMarket is closed. Orders will queue for the next open "
              f"({clock['next_open']}).")

    print("\nSubmitting orders...")
    for sym, r in plan.iterrows():
        try:
            order = client.trading.submit_order(
                MarketOrderRequest(
                    symbol=sym,
                    qty=int(r["shares"]),
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            )
            print(f"  {sym:6s} BUY {int(r['shares']):>5} -> {order.status}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {sym:6s} FAILED: {type(exc).__name__}: {exc}")

    print("\nDone. Re-run scripts/preflight.py to see the live risk report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
