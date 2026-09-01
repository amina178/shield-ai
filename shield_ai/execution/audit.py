"""
Shield-AI — evidence layer.

A trading agent that cannot prove what it did is a story, not a system. This
module produces three artifacts a judge can verify independently:

  1. decisions.jsonl — every decision the agent reached, INCLUDING the ones
     where it chose not to trade and why. For a risk agent this is the most
     important record of all: a premium sale declined because of an earnings
     veto is direct evidence that the guardrails work. A log that only contains
     executed trades proves the agent traded; a log that contains refusals
     proves the agent had judgement.

  2. trades.jsonl — the live execution ledger, carrying the broker-assigned
     order ID for every order. That ID can be reconciled against the Alpaca
     account, which makes it evidence rather than a claim.

  3. tearsheet — realised P&L with risk-adjusted context: Sharpe, Sortino,
     maximum drawdown, profit factor, win rate. Never a win rate alone; a
     premium-selling strategy wins often and small and loses rarely and large,
     so a bare win rate flatters the strategy and tells the reader nothing.

Both logs are append-only JSON Lines. One self-contained JSON object per line
means a crash mid-write can corrupt at most the final record, and the file
stays readable by pandas, jq, or a human.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

TRADING_DAYS = 252

Decision = Literal["SELL_PREMIUM", "BUY_PROTECTION", "HEDGE", "CLOSE", "HOLD"]
Outcome = Literal["EXECUTED", "REJECTED", "VETOED", "NO_ACTION", "FAILED"]


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


@dataclass
class DecisionRecord:
    """One complete pass of the agent's reasoning, whatever the conclusion."""

    timestamp: str
    cycle_id: str
    symbol: str
    decision: Decision
    outcome: Outcome

    # The quantitative state that triggered the decision
    var_headline: float | None = None
    var_limit: float | None = None
    net_delta: float | None = None
    implied_vol: float | None = None
    forecast_vol: float | None = None
    edge_points: float | None = None

    # Why it ended the way it did — the part that carries the evidence
    reason: str = ""
    rejected_by: str = ""          # which gate blocked it, if any
    llm_rationale: str = ""        # the cognitive layer's own words
    candidate: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


@dataclass
class TradeRecord:
    """One order actually sent to the broker."""

    timestamp: str
    cycle_id: str
    broker_order_id: str           # the unfakeable part
    symbol: str
    side: str
    qty: float
    order_type: str
    limit_price: float | None
    asset_class: str
    strategy: str
    status: str
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


class AuditLog:
    """Append-only writer for both records. Never overwrites, never rewrites."""

    def __init__(self, directory: str | Path = "logs") -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.decisions_path = self.dir / "decisions.jsonl"
        self.trades_path = self.dir / "trades.jsonl"

    @staticmethod
    def new_cycle_id() -> str:
        """Identifier tying every record produced by one agent pass together."""
        return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def _append(self, path: Path, line: str) -> None:
        # Opened per write and flushed immediately: an agent that dies holding a
        # buffer loses exactly the evidence you most wanted to keep.
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def log_decision(self, record: DecisionRecord) -> None:
        self._append(self.decisions_path, record.to_json())

    def log_trade(self, record: TradeRecord) -> None:
        self._append(self.trades_path, record.to_json())

    # -- reading back -------------------------------------------------------

    def decisions(self) -> pd.DataFrame:
        return self._read(self.decisions_path)

    def trades(self) -> pd.DataFrame:
        return self._read(self.trades_path)

    @staticmethod
    def _read(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        rows = []
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A truncated final line from an interrupted write. Skip it and
                # keep every complete record before it.
                print(f"  warning: skipping malformed line {i} in {path.name}")
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tearsheet
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tearsheet:
    """Performance with the risk context that makes a return number meaningful."""

    start_equity: float
    end_equity: float
    total_return: float
    n_days: int

    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    volatility_annual: float

    n_trades: int
    win_rate: float
    profit_factor: float
    largest_win: float
    largest_loss: float

    def summary(self) -> str:
        return (
            f"Equity      ${self.start_equity:,.0f} -> ${self.end_equity:,.0f} "
            f"({self.total_return:+.2%}) over {self.n_days} days\n"
            f"Sharpe      {self.sharpe:6.2f}\n"
            f"Sortino     {self.sortino:6.2f}\n"
            f"Max DD      {self.max_drawdown:6.2%}\n"
            f"Calmar      {self.calmar:6.2f}\n"
            f"Volatility  {self.volatility_annual:6.2%} annualised\n"
            f"Trades      {self.n_trades}\n"
            f"Win rate    {self.win_rate:6.1%}  "
            f"(interpret only alongside largest loss)\n"
            f"Profit factor {self.profit_factor:.2f}\n"
            f"Largest win  ${self.largest_win:,.2f}\n"
            f"Largest loss ${self.largest_loss:,.2f}"
        )


def build_tearsheet(
    equity_curve: pd.Series,
    trade_pnl: pd.Series | None = None,
    risk_free_rate: float = 0.04,
) -> Tearsheet:
    """
    Compute performance statistics from an equity curve.

    `equity_curve` is account equity indexed by date. `trade_pnl` is realised
    profit per closed trade, used only for the trade-level statistics.

    Sharpe uses excess return over the risk-free rate — omitting it inflates the
    ratio, which matters when a strategy's whole edge is a few percent a year.
    Sortino divides by DOWNSIDE deviation only, which is the fairer measure for
    an asymmetric premium-selling payoff: upside variance is not a risk.
    """
    eq = pd.Series(equity_curve).dropna().astype(float)
    if len(eq) < 2:
        raise ValueError("Need at least two equity observations.")

    rets = eq.pct_change().dropna()
    n_days = len(rets)

    daily_rf = risk_free_rate / TRADING_DAYS
    excess = rets - daily_rf

    vol_daily = float(rets.std(ddof=1))
    vol_annual = vol_daily * np.sqrt(TRADING_DAYS)

    sharpe = (
        float(excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if excess.std(ddof=1) > 0 else 0.0
    )

    downside = excess[excess < 0]
    dd_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = (
        float(excess.mean() / dd_std * np.sqrt(TRADING_DAYS)) if dd_std > 0 else 0.0
    )

    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0
    max_dd = float(drawdown.min())

    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    years = max(n_days / TRADING_DAYS, 1e-9)
    annualised = (1.0 + total_return) ** (1.0 / years) - 1.0
    calmar = float(annualised / abs(max_dd)) if max_dd < 0 else 0.0

    if trade_pnl is not None and len(trade_pnl):
        p = pd.Series(trade_pnl).dropna().astype(float)
        wins, losses = p[p > 0], p[p < 0]
        win_rate = float(len(wins) / len(p))
        gross_loss = float(-losses.sum())
        profit_factor = (
            float(wins.sum() / gross_loss) if gross_loss > 0
            else float("inf") if len(wins) else 0.0
        )
        largest_win = float(wins.max()) if len(wins) else 0.0
        largest_loss = float(losses.min()) if len(losses) else 0.0
        n_trades = int(len(p))
    else:
        win_rate = profit_factor = largest_win = largest_loss = 0.0
        n_trades = 0

    return Tearsheet(
        start_equity=float(eq.iloc[0]),
        end_equity=float(eq.iloc[-1]),
        total_return=total_return,
        n_days=n_days,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
        volatility_annual=vol_annual,
        n_trades=n_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        largest_win=largest_win,
        largest_loss=largest_loss,
    )
