"""
Shield-AI — preflight integration check.

Run this BEFORE the market opens. It answers one question with evidence:
"is my code correctly wired to Alpaca?" — separately from "does my strategy
work", which is the only part that actually needs a live session.

    PYTHONPATH=. python3 scripts/preflight.py

Every check below works with the market closed.
"""

from __future__ import annotations

import sys
import time

# Printed BEFORE the heavy imports on purpose. pandas, arch and their compiled
# dependencies can take a long time to load the first time on a machine with
# endpoint security software that scans freshly installed binaries. Without this
# line the screen stays blank during that load and the script looks hung when it
# is merely importing.
print("Shield-AI preflight — loading libraries...", flush=True)
_t0 = time.monotonic()

import traceback

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("note: python-dotenv not installed; relying on exported env vars.\n")

from shield_ai.config import MANDATE
from shield_ai.data.alpaca_client import ShieldDataClient, AlpacaConfigError
from shield_ai.quant.risk import build_risk_report
from shield_ai.quant.garch import forecast_volatility, compute_edge

OK = "  \033[32mPASS\033[0m"
BAD = "  \033[31mFAIL\033[0m"
WARN = "  \033[33mWARN\033[0m"

results: list[tuple[str, bool]] = []


def check(name: str):
    """Decorator turning a function into a reported check."""
    def wrap(fn):
        def run(*a, **kw):
            print(f"\n[{name}]")
            try:
                out = fn(*a, **kw)
                results.append((name, True))
                return out
            except Exception as exc:  # noqa: BLE001
                print(f"{BAD} {type(exc).__name__}: {exc}")
                if "--verbose" in sys.argv:
                    traceback.print_exc()
                results.append((name, False))
                return None
        return run
    return wrap


@check("1. Credentials and account")
def check_account(client: ShieldDataClient):
    acct = client.account()
    print(f"{OK} connected — account status: {acct.status}")
    print(f"       equity           ${acct.equity:,.2f}")
    print(f"       cash             ${acct.cash:,.2f}")
    print(f"       buying power     ${acct.buying_power:,.2f}")
    print(f"       options BP       ${acct.options_buying_power:,.2f}")
    print(f"       options level    {acct.options_trading_level} "
          f"(approved: {acct.options_approved_level})")

    if acct.equity < 1000:
        print(f"{WARN} equity looks unexpectedly low for a paper account")

    # The blocker that must be resolved before Monday.
    if not acct.can_buy_puts:
        print(f"{BAD} options level {acct.options_trading_level}: long puts need "
              f"LEVEL 2. The hedging leg cannot execute. Request an upgrade now.")
    elif not acct.can_trade_multileg:
        print(f"{WARN} level {acct.options_trading_level}: single-leg puts OK, but "
              f"multi-leg collars need LEVEL 3.")
    else:
        print(f"{OK} level 3 — protective puts and multi-leg collars both available")
    return acct


@check("2. Market clock")
def check_clock(client: ShieldDataClient):
    clock = client.market_clock()
    state = "OPEN" if clock["is_open"] else "CLOSED"
    print(f"{OK} market is {state}")
    print(f"       next open        {clock['next_open']}")
    print(f"       next close       {clock['next_close']}")
    if not clock["is_open"]:
        print("       (expected outside session hours — every check below still runs)")
    return clock


@check("3. Historical bars — the covariance matrix input")
def check_bars(client: ShieldDataClient):
    symbols = list(MANDATE.universe)
    closes = client.daily_closes(symbols, lookback_days=400)
    print(f"{OK} fetched {closes.shape[0]} daily closes x {closes.shape[1]} symbols")
    print(f"       range            {closes.index[0].date()} -> {closes.index[-1].date()}")

    if closes.shape[0] < MANDATE.lookback_days:
        print(f"{WARN} only {closes.shape[0]} rows; mandate wants "
              f"{MANDATE.lookback_days}. Increase lookback_days.")

    missing = closes.isna().sum()
    if missing.any():
        print(f"{WARN} missing values: {missing[missing > 0].to_dict()}")

    print("\n       latest closes:")
    for sym, px in closes.iloc[-1].items():
        print(f"         {sym:6s} ${px:,.2f}")
    return closes


@check("4. Current positions")
def check_positions(client: ShieldDataClient):
    pos = client.positions()
    opts = client.option_positions()
    if pos.empty:
        print(f"{WARN} no equity positions yet — run scripts/seed_portfolio.py")
    else:
        print(f"{OK} {len(pos)} equity positions, "
              f"${pos.abs().sum():,.2f} gross exposure")
        for sym, val in pos.items():
            print(f"         {sym:6s} ${val:>12,.2f}")
    print(f"       option positions : {len(opts)}")
    return pos


@check("5. News feed — the Sentiment Agent input")
def check_news(client: ShieldDataClient):
    items = client.latest_news(list(MANDATE.universe), hours_back=72, limit=10)
    print(f"{OK} {len(items)} headlines in the last 72h")
    for n in items[:5]:
        syms = ",".join(n["symbols"][:3])
        print(f"         [{syms:14s}] {n['headline'][:70]}")
    if not items:
        print(f"{WARN} empty news feed — verify your data subscription")
    return items


@check("6. Option chain and Greeks")
def check_options(client: ShieldDataClient):
    underlying = MANDATE.benchmark
    contracts = client.option_contracts(
        underlying, min_dte=MANDATE.min_dte, max_dte=MANDATE.max_dte
    )
    print(f"{OK} {len(contracts)} put contracts on {underlying} "
          f"in the {MANDATE.min_dte}-{MANDATE.max_dte} DTE window")
    if not contracts:
        print(f"{WARN} no contracts — widen the DTE window or check the symbol")
        return None

    sample = [c["symbol"] for c in contracts[:20]]
    snaps = client.option_snapshots(sample)
    print(f"{OK} {len(snaps)} of {len(sample)} sampled contracts returned IV")

    shown = 0
    for c in contracts[:20]:
        s = snaps.get(c["symbol"])
        if not s:
            continue
        delta_txt = f"{s['delta']:+.3f}" if s["delta"] is not None else "n/a"
        print(f"         {c['symbol']:22s} K={c['strike']:>7.1f} "
              f"dte={c['dte']:>2d} IV={s['implied_volatility']:.1%} "
              f"delta={delta_txt}")
        shown += 1
        if shown >= 5:
            break

    if not snaps:
        print(f"{WARN} no IV returned. On the free 'indicative' feed this is "
              f"common outside market hours — recheck after Monday's open.")
    return snaps


@check("7. Risk engine on live data")
def check_risk(client: ShieldDataClient, closes: pd.DataFrame, positions: pd.Series):
    if closes is None:
        raise RuntimeError("no price data — check 3 must pass first")

    acct = client.account()
    if positions is None or positions.empty:
        # Hypothetical equal-weight book, so the engine can be validated before
        # any capital is deployed.
        n = len(MANDATE.universe)
        positions = pd.Series(
            {s: acct.equity * 0.5 / n for s in MANDATE.universe}, dtype=float
        )
        print("       using a hypothetical 50%-invested equal-weight book")

    report = build_risk_report(
        closes,
        positions,
        equity=acct.equity,
        confidence=MANDATE.confidence,
        student_t_df=MANDATE.student_t_df,
        mc_paths=20_000,
        benchmark=MANDATE.benchmark,
        min_observations=MANDATE.min_observations,
    )
    print(f"{OK} risk computed on {report.n_observations} observations")
    print(f"       VaR parametric   {report.var_parametric:.3%}")
    print(f"       VaR historical   {report.var_historical:.3%}")
    print(f"       VaR monte-carlo  {report.var_monte_carlo:.3%}")
    print(f"       VaR headline     {report.var_headline:.3%} "
          f"(${report.var_dollars:,.0f})")
    print(f"       CVaR             {report.cvar:.3%}")
    print(f"       net delta        {report.net_delta:.3f}")
    print(f"       disagreement     {report.model_disagreement:.1%}")

    limit = MANDATE.var_limit_pct
    if report.var_headline > limit:
        print(f"       -> BREACH: {report.var_headline:.2%} exceeds the "
              f"{limit:.2%} mandate. The agent would hedge.")
    else:
        print(f"       -> within mandate ({limit:.2%})")
    return report


@check("8. GARCH forecast and edge signal")
def check_garch(closes: pd.DataFrame):
    if closes is None:
        raise RuntimeError("no price data — check 3 must pass first")
    import numpy as np

    sym = MANDATE.benchmark
    rets = np.log(closes[sym] / closes[sym].shift(1)).dropna()
    fc = forecast_volatility(rets, symbol=sym, horizon_days=14)

    print(f"{OK} GARCH(1,1) fitted on {len(rets)} returns of {sym}")
    print(f"       alpha            {fc.alpha:.4f}")
    print(f"       beta             {fc.beta:.4f}")
    print(f"       persistence      {fc.persistence:.4f}")
    print(f"       14d vol forecast {fc.annualized_vol:.2%} annualised")

    if not fc.is_trustworthy:
        print(f"{WARN} model did not converge or is non-stationary — do not "
              f"trade on this forecast")

    # Show where the decision boundary sits today.
    for iv in (0.15, 0.20, 0.25, 0.30):
        edge = compute_edge(iv, fc)
        print(f"       IV {iv:.0%} -> edge {edge.edge_points:+6.1f} pts -> "
              f"{edge.signal(MANDATE.min_edge_vol_points)}")
    return fc


def main() -> int:
    print(f"libraries loaded in {time.monotonic() - _t0:.1f}s\n")
    print("=" * 66)
    print("Shield-AI preflight — verifying Alpaca integration")
    print("=" * 66)

    try:
        client = ShieldDataClient(paper=True)
    except AlpacaConfigError as exc:
        print(f"\n{BAD} {exc}")
        return 1

    check_account(client)
    check_clock(client)
    closes = check_bars(client)
    positions = check_positions(client)
    check_news(client)
    check_options(client)
    check_risk(client, closes, positions)
    check_garch(closes)

    passed = sum(1 for _, ok in results if ok)
    print("\n" + "=" * 66)
    print(f"{passed}/{len(results)} checks passed")
    print("=" * 66)
    if passed < len(results):
        print("\nRe-run with --verbose for full tracebacks.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
