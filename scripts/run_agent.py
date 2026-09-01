"""
Shield-AI — the agent cycle.

One pass: read the book, measure the risk, forecast volatility, compare it to
what options are implying, ask the news layer whether anything is about to blow
up, and either act or record why not.

    PYTHONPATH=. python3 -u scripts/run_agent.py            # dry run
    PYTHONPATH=. python3 -u scripts/run_agent.py --execute  # place orders
    PYTHONPATH=. python3 -u scripts/run_agent.py --loop 15  # every 15 minutes

Every cycle writes to logs/decisions.jsonl whether or not it trades. That file
is the deliverable: it shows the agent reasoning on days it did nothing, which
is what distinguishes a risk system from a trade generator.
"""

from __future__ import annotations

import argparse
import sys
import time

print("Shield-AI agent — loading libraries...", flush=True)
_t0 = time.monotonic()

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from alpaca.trading.enums import ContractType, OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest

from shield_ai.config import MANDATE
from shield_ai.data.alpaca_client import ShieldDataClient, AlpacaConfigError
from shield_ai.quant.risk import build_risk_report, log_returns
from shield_ai.quant.garch import forecast_volatility, compute_edge
from shield_ai.agents.sentiment import assess
from shield_ai.agents.risk_manager import decide_action, build_proposal
from shield_ai.execution import policy
from shield_ai.execution.audit import (
    AuditLog, DecisionRecord, TradeRecord, _utc_now,
)


def _fmt(x: float | None, spec: str = ".2f") -> str:
    return format(x, spec) if x is not None else "n/a"


def run_cycle(client: ShieldDataClient, log: AuditLog, execute: bool) -> dict:
    """One complete pass of the agent. Returns a small summary dict."""
    cycle = log.new_cycle_id()
    print(f"\n{'=' * 70}")
    print(f"CYCLE {cycle}   mode={'EXECUTE' if execute else 'DRY RUN'}")
    print("=" * 70)

    acct = client.account()
    clock = client.market_clock()
    positions = client.positions()

    print(f"equity ${acct.equity:,.2f}   cash ${acct.cash:,.2f}   "
          f"market {'OPEN' if clock['is_open'] else 'CLOSED'}")

    if positions.empty:
        print("No equity positions — run scripts/seed_portfolio.py first.")
        return {"cycle": cycle, "acted": 0, "considered": 0}

    # ---- 1. Deterministic risk pass ------------------------------------
    universe = list(MANDATE.universe)
    closes = client.daily_closes(universe, lookback_days=400)
    report = build_risk_report(
        closes, positions, equity=acct.equity,
        confidence=MANDATE.confidence, student_t_df=MANDATE.student_t_df,
        mc_paths=MANDATE.mc_paths, benchmark=MANDATE.benchmark,
        min_observations=MANDATE.min_observations,
    )

    print(f"\nRISK   VaR {report.var_headline:.3%} (${report.var_dollars:,.0f})"
          f"  limit {MANDATE.var_limit_pct:.2%}"
          f"   CVaR {report.cvar:.3%}"
          f"   delta {report.net_delta:.3f}"
          f"   disagreement {report.model_disagreement:.1%}")

    # ---- 2. News, once for the whole universe --------------------------
    news = client.latest_news(universe, hours_back=72, limit=50)
    print(f"NEWS   {len(news)} headlines in the last 72h")

    rets = log_returns(closes)
    considered = acted = 0

    # ---- 3. Per-symbol decision ----------------------------------------
    for symbol in universe:
        considered += 1
        print(f"\n--- {symbol} " + "-" * (66 - len(symbol)))

        # Volatility forecast over the tenor we would actually trade.
        horizon = (MANDATE.min_dte + MANDATE.max_dte) // 2
        try:
            fc = forecast_volatility(rets[symbol], symbol=symbol,
                                     horizon_days=horizon)
        except Exception as exc:  # noqa: BLE001
            print(f"  GARCH failed: {exc}")
            continue

        if not fc.is_trustworthy:
            print(f"  GARCH not trustworthy (persistence {fc.persistence:.3f})"
                  f" — skipping")
            log.log_decision(DecisionRecord(
                timestamp=_utc_now(), cycle_id=cycle, symbol=symbol,
                decision="HOLD", outcome="NO_ACTION",
                var_headline=report.var_headline, forecast_vol=fc.annualized_vol,
                reason="GARCH model not stationary or did not converge",
                rejected_by="model_quality",
            ))
            continue

        # Options chain. Puts on both legs: a short put is the income leg
        # (cash-secured), a long put is the hedge.
        contracts = client.option_contracts(
            symbol, min_dte=MANDATE.min_dte, max_dte=MANDATE.max_dte,
            contract_type=ContractType.PUT,
        )
        if not contracts:
            print(f"  no put contracts in the {MANDATE.min_dte}-"
                  f"{MANDATE.max_dte} DTE window")
            continue

        snaps = client.option_snapshots([c["symbol"] for c in contracts[:60]])
        if not snaps:
            print("  no implied volatility returned — chain unusable")
            continue

        # Reference IV: the contract closest to our target delta.
        atm_iv = float(np.median([s["implied_volatility"] for s in snaps.values()]))
        edge = compute_edge(atm_iv, fc)

        print(f"  IV {atm_iv:.1%}   GARCH {fc.annualized_vol:.1%}   "
              f"edge {edge.edge_points:+.1f} pts   "
              f"{len(contracts)} contracts, {len(snaps)} priced")

        action, why = decide_action(
            report.var_headline, report.net_delta, edge.edge_points, MANDATE
        )
        print(f"  DECISION {action}: {why}")

        # ---- 4. Sentiment veto, before any pricing work ------------------
        verdict = assess(symbol, news, horizon_days=horizon)
        veto = ""
        if action == "SELL_PREMIUM" and verdict.blocks_premium_selling:
            veto = verdict.veto_reason
            print(f"  VETO ({verdict.source}): {veto}")

        proposal = build_proposal(
            action=action, rationale=why, symbol=symbol,
            contracts=contracts, snapshots=snaps, equity=acct.equity,
            var_headline=report.var_headline, mandate=MANDATE,
        )

        base = dict(
            timestamp=_utc_now(), cycle_id=cycle, symbol=symbol,
            var_headline=report.var_headline, var_limit=MANDATE.var_limit_pct,
            net_delta=report.net_delta, implied_vol=atm_iv,
            forecast_vol=fc.annualized_vol, edge_points=edge.edge_points,
            llm_rationale=verdict.rationale,
        )

        if veto:
            log.log_decision(DecisionRecord(
                **base, decision=action, outcome="VETOED",
                reason=why, rejected_by="sentiment_veto",
                candidate={"veto": veto, "evidence": verdict.evidence},
            ))
            continue

        if proposal.action == "HOLD":
            print(f"  -> HOLD ({proposal.rationale})")
            log.log_decision(DecisionRecord(
                **base, decision="HOLD", outcome="NO_ACTION",
                reason=proposal.rationale,
            ))
            continue

        # ---- 5. Deterministic policy gates ------------------------------
        shares_held = int(positions.get(symbol, 0.0) /
                          float(closes[symbol].iloc[-1]))
        gate = policy.evaluate(
            strategy=proposal.strategy, side=proposal.side,
            contract=proposal.contract, snapshot=proposal.snapshot,
            contracts=proposal.contracts, equity=acct.equity,
            cash_available=acct.cash,
            options_buying_power=acct.options_buying_power,
            shares_held=shares_held,
            options_trading_level=acct.options_trading_level,
            mandate=MANDATE,
        )

        if not gate.allowed:
            print(f"  -> BLOCKED by {gate.gate}: {gate.reason}")
            log.log_decision(DecisionRecord(
                **base, decision=proposal.action, outcome="REJECTED",
                reason=proposal.rationale, rejected_by=gate.gate,
                candidate={"contract": proposal.contract["symbol"],
                           "gate_reason": gate.reason},
            ))
            continue

        print(f"  -> APPROVED {proposal.strategy} {proposal.side} "
              f"{proposal.contracts}x {proposal.contract['symbol']} "
              f"@ {_fmt(proposal.limit_price)}")

        if not execute:
            log.log_decision(DecisionRecord(
                **base, decision=proposal.action, outcome="NO_ACTION",
                reason=proposal.rationale,
                rejected_by="dry_run",
                candidate={"contract": proposal.contract["symbol"],
                           "contracts": proposal.contracts,
                           "limit": proposal.limit_price},
            ))
            continue

        # ---- 6. Execute ------------------------------------------------
        try:
            order = client.trading.submit_order(
                LimitOrderRequest(
                    symbol=proposal.contract["symbol"],
                    qty=proposal.contracts,
                    side=OrderSide.SELL if proposal.side == "SELL" else OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    limit_price=proposal.limit_price,
                )
            )
            print(f"     order {order.id} -> {order.status}")
            acted += 1

            log.log_decision(DecisionRecord(
                **base, decision=proposal.action, outcome="EXECUTED",
                reason=proposal.rationale,
                candidate={"contract": proposal.contract["symbol"],
                           "order_id": str(order.id)},
            ))
            log.log_trade(TradeRecord(
                timestamp=_utc_now(), cycle_id=cycle,
                broker_order_id=str(order.id),
                symbol=proposal.contract["symbol"],
                side=proposal.side, qty=proposal.contracts,
                order_type="limit", limit_price=proposal.limit_price,
                asset_class="option", strategy=proposal.strategy,
                status=str(order.status), notes=why,
            ))
        except Exception as exc:  # noqa: BLE001
            print(f"     ORDER FAILED: {type(exc).__name__}: {exc}")
            log.log_decision(DecisionRecord(
                **base, decision=proposal.action, outcome="FAILED",
                reason=proposal.rationale, rejected_by="broker",
                candidate={"error": f"{type(exc).__name__}: {exc}"[:200]},
            ))

    print(f"\n{'=' * 70}")
    print(f"cycle {cycle}: {considered} symbols considered, {acted} order(s) placed")
    print(f"decisions logged to {log.decisions_path}")
    return {"cycle": cycle, "acted": acted, "considered": considered}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true",
                    help="place real orders (default is a dry run)")
    ap.add_argument("--loop", type=int, default=0, metavar="MINUTES",
                    help="repeat every N minutes until interrupted")
    args = ap.parse_args()

    print(f"libraries loaded in {time.monotonic() - _t0:.1f}s")

    try:
        client = ShieldDataClient(paper=True)
    except AlpacaConfigError as exc:
        print(f"ERROR: {exc}")
        return 1

    log = AuditLog("logs")

    if args.loop <= 0:
        run_cycle(client, log, args.execute)
        return 0

    print(f"Looping every {args.loop} minute(s). Ctrl+C to stop.")
    try:
        while True:
            try:
                run_cycle(client, log, args.execute)
            except Exception as exc:  # noqa: BLE001
                # One bad cycle must never kill the agent. A trading system that
                # dies on a transient API error is worse than one that skips a
                # cycle, because it stops managing the risk it already carries.
                print(f"\nCYCLE FAILED: {type(exc).__name__}: {exc}")
            print(f"\nsleeping {args.loop} min...")
            time.sleep(args.loop * 60)
    except KeyboardInterrupt:
        print("\nstopped by user")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
