"""
Shield-AI — Risk-Manager Agent.

Turns a signal into a concrete, sized, priced order proposal. It decides WHAT to
trade; the policy filter decides whether that proposal may reach the market, and
this module never bypasses it.

Two responsibilities:

  1. Decide the action from the quantitative state (VaR breach, edge sign).
  2. Pick the specific contract and size it against the remaining risk budget.

Sizing is solved for, never chosen. The question is never "how many contracts do
I want" but "how many contracts fit inside what is left of the mandate".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shield_ai.config import RiskMandate
from shield_ai.execution.policy import midpoint, spread_pct


@dataclass(frozen=True)
class Proposal:
    """A fully specified order the policy filter can accept or refuse."""

    action: str                     # SELL_PREMIUM | BUY_PROTECTION | HEDGE | HOLD
    strategy: str                   # covered_call | cash_secured_put | protective_put | collar
    symbol: str                     # underlying
    contract: dict = field(default_factory=dict)
    snapshot: dict = field(default_factory=dict)
    side: str = ""                  # BUY | SELL
    contracts: int = 0
    limit_price: float | None = None
    rationale: str = ""

    @property
    def notional(self) -> float:
        return (self.limit_price or 0.0) * 100.0 * self.contracts


def decide_action(
    var_headline: float,
    net_delta: float,
    edge_points: float,
    mandate: RiskMandate,
) -> tuple[str, str]:
    """
    The deterministic decision tree. Returns (action, rationale).

    Order of precedence is the whole thesis in five lines: risk first, income
    second. A VaR breach outranks any income opportunity, however rich the
    premium — because a breach means the book can already lose more than the
    mandate allows, and adding a short option to that is adding risk to a
    position that is already over budget.
    """
    if var_headline > mandate.var_limit_pct:
        return (
            "HEDGE",
            f"VaR {var_headline:.2%} exceeds mandate {mandate.var_limit_pct:.2%}",
        )

    if net_delta > mandate.delta_band[1]:
        return (
            "HEDGE",
            f"net delta {net_delta:.2f} above band top {mandate.delta_band[1]:.2f}",
        )

    if edge_points <= -mandate.min_edge_vol_points:
        return (
            "BUY_PROTECTION",
            f"IV {abs(edge_points):.1f} vol points BELOW forecast — "
            f"protection is cheap",
        )

    if edge_points >= mandate.min_edge_vol_points:
        if var_headline > mandate.var_warn_pct:
            return (
                "HOLD",
                f"edge {edge_points:+.1f} pts is attractive but VaR "
                f"{var_headline:.2%} is already in the warn zone "
                f"({mandate.var_warn_pct:.2%})",
            )
        return (
            "SELL_PREMIUM",
            f"IV {edge_points:.1f} vol points ABOVE forecast — premium is rich",
        )

    return (
        "HOLD",
        f"edge {edge_points:+.1f} pts inside the "
        f"±{mandate.min_edge_vol_points:.0f} pt no-trade band",
    )


def select_contract(
    contracts: list[dict],
    snapshots: dict[str, dict],
    target_delta: float,
    prefer_put: bool = True,
) -> tuple[dict, dict] | None:
    """
    Choose the contract whose delta is closest to the target.

    Delta is used as the selection axis rather than strike distance because
    delta is comparable across underlyings and volatility regimes: a 20-delta
    option is roughly a 20% chance of finishing in the money whether the name is
    a utility or a semiconductor. Picking "5% out of the money" instead would
    mean something completely different on each.

    Contracts without a usable quote are skipped here rather than passed on —
    an unpriceable contract cannot be sized, and letting it through would only
    move the failure further down the pipeline.
    """
    best: tuple[float, dict, dict] | None = None

    for c in contracts:
        snap = snapshots.get(c["symbol"])
        if not snap:
            continue
        delta = snap.get("delta")
        if delta is None:
            continue
        if midpoint(snap.get("bid"), snap.get("ask")) is None:
            continue

        # Put deltas are negative; compare on magnitude.
        distance = abs(abs(delta) - abs(target_delta))
        if best is None or distance < best[0]:
            best = (distance, c, snap)

    if best is None:
        return None
    return best[1], best[2]


def size_by_risk_budget(
    limit_price: float,
    equity: float,
    var_headline: float,
    mandate: RiskMandate,
    max_contracts: int = 10,
) -> int:
    """
    Number of contracts that fits the remaining risk budget.

    Headroom is what is left between current VaR and the limit. We allow a
    position whose worst plausible cost consumes at most half of that headroom,
    so a single trade can never take the book from compliant to breached — the
    agent must always be able to hedge after acting.
    """
    headroom = max(mandate.var_limit_pct - var_headline, 0.0)
    if headroom <= 0:
        return 0

    budget_dollars = equity * headroom * 0.5
    cost_per_contract = limit_price * 100.0
    if cost_per_contract <= 0:
        return 0

    n = int(budget_dollars // cost_per_contract)
    return max(0, min(n, max_contracts, mandate.max_positions))


def build_proposal(
    *,
    action: str,
    rationale: str,
    symbol: str,
    contracts: list[dict],
    snapshots: dict[str, dict],
    equity: float,
    var_headline: float,
    mandate: RiskMandate,
) -> Proposal:
    """Assemble a complete, priced, sized proposal for the policy filter."""
    if action == "HOLD":
        return Proposal(action="HOLD", strategy="", symbol=symbol,
                        rationale=rationale)

    picked = select_contract(contracts, snapshots, mandate.target_short_delta)
    if picked is None:
        return Proposal(
            action="HOLD", strategy="", symbol=symbol,
            rationale=f"{rationale}; but no contract had a usable delta and quote",
        )

    contract, snapshot = picked
    price = midpoint(snapshot.get("bid"), snapshot.get("ask"))

    if action in ("HEDGE", "BUY_PROTECTION"):
        strategy, side = "protective_put", "BUY"
    else:
        strategy = "cash_secured_put" if contract.get("type") == "put" else "covered_call"
        side = "SELL"

    n = size_by_risk_budget(price, equity, var_headline, mandate)
    if n == 0:
        return Proposal(
            action="HOLD", strategy=strategy, symbol=symbol,
            contract=contract, snapshot=snapshot,
            rationale=f"{rationale}; but the risk budget allows 0 contracts",
        )

    sp = spread_pct(snapshot.get("bid"), snapshot.get("ask"))
    return Proposal(
        action=action,
        strategy=strategy,
        symbol=symbol,
        contract=contract,
        snapshot=snapshot,
        side=side,
        contracts=n,
        limit_price=price,
        rationale=(
            f"{rationale}; selected {contract['symbol']} "
            f"(strike {contract['strike']}, {contract['dte']}d, "
            f"delta {snapshot.get('delta'):+.3f}, IV "
            f"{snapshot.get('implied_volatility', 0):.1%}, spread "
            f"{sp:.1%}) x{n} at mid {price:.2f}"
        ),
    )
