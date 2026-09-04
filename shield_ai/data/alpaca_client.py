"""
Shield-AI — Alpaca data and account access.

This is the READ side of the system. Per Alpaca's own engineering guidance,
market data and the trading loop run through `alpaca-py` (full control, no
round-trip through a protocol layer), while ORDER PLACEMENT goes through the
MCP server in `shield_ai/execution/`. Keeping the two separate is deliberate:
the quant engine needs a 252-day covariance matrix, which is a bulk data
problem, not an agent-tool problem.

Everything here works with the market closed. Historical bars, account state,
news and option contract metadata are all available 24/7 — only live quotes and
order fills need an open session. That is what lets the whole stack be debugged
over a weekend.
"""

from __future__ import annotations

import datetime as dt
import os
import re
from dataclasses import dataclass

import pandas as pd
from alpaca.data.enums import DataFeed, OptionsFeed
from alpaca.data.historical.news import NewsClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import (
    NewsRequest,
    OptionSnapshotRequest,
    StockBarsRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetStatus, ContractType
from alpaca.trading.requests import GetOptionContractsRequest


class AlpacaConfigError(RuntimeError):
    """Raised when credentials are missing or the account is not what we expect."""


def _is_option(position) -> bool:
    """
    True when a broker position is an option rather than a stock.

    Alpaca reports the class as the enum `AssetClass.US_OPTION`, so `str()` of it
    is "AssetClass.US_OPTION" — UPPERCASE. An earlier version of this check tested
    `.endswith("option")` in lowercase and therefore never matched, which let every
    option contract fall through into the equity book. The risk engine then treated
    it as a linear position and beta-weighted it like a stock, which is wrong twice
    over: an option's exposure is its delta, not its market value, and that delta
    moves with the underlying.

    Matching on the OCC symbol shape as well makes the check independent of how the
    SDK spells the enum: an OCC contract symbol is the underlying followed by
    6 digits of date, a C or P, and 8 digits of strike — at least 15 characters
    ending in exactly that pattern.
    """
    cls = str(getattr(position, "asset_class", "")).lower()
    if "option" in cls:
        return True
    sym = str(getattr(position, "symbol", ""))
    return bool(re.fullmatch(r"[A-Z]{1,6}\d{6}[CP]\d{8}", sym))


@dataclass(frozen=True)
class AccountSnapshot:
    """The account facts the risk engine and policy filter need."""

    equity: float
    cash: float
    buying_power: float
    options_buying_power: float
    options_approved_level: int
    options_trading_level: int
    pattern_day_trader: bool
    status: str

    @property
    def can_buy_puts(self) -> bool:
        """Long puts require options Level 2."""
        return self.options_trading_level >= 2

    @property
    def can_trade_multileg(self) -> bool:
        """Collars and other spreads require options Level 3."""
        return self.options_trading_level >= 3


class ShieldDataClient:
    """
    One object that owns every read-only connection to Alpaca.

    Feeds default to the free tier (IEX for stocks, indicative for options).
    If you have a paid market-data subscription, pass feed=DataFeed.SIP and
    options_feed=OptionsFeed.OPRA for full-market coverage. Using SIP without a
    subscription produces a confusing 'subscription does not permit' error
    rather than an obvious auth failure, which is why IEX is the default here.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
        feed: DataFeed = DataFeed.IEX,
        options_feed: OptionsFeed = OptionsFeed.INDICATIVE,
    ) -> None:
        self.api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "")

        if not self.api_key or not self.secret_key:
            raise AlpacaConfigError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY are not set. "
                "Copy .env.example to .env and fill them in, then load it with "
                "python-dotenv or `export $(grep -v '^#' .env | xargs)`."
            )

        self.paper = paper
        self.feed = feed
        self.options_feed = options_feed

        self.trading = TradingClient(self.api_key, self.secret_key, paper=paper)
        self.stocks = StockHistoricalDataClient(self.api_key, self.secret_key)
        self.options = OptionHistoricalDataClient(self.api_key, self.secret_key)
        self.news_client = NewsClient(self.api_key, self.secret_key)

    # -- account ------------------------------------------------------------

    def account(self) -> AccountSnapshot:
        a = self.trading.get_account()
        return AccountSnapshot(
            equity=float(a.equity or 0),
            cash=float(a.cash or 0),
            buying_power=float(a.buying_power or 0),
            options_buying_power=float(getattr(a, "options_buying_power", 0) or 0),
            options_approved_level=int(getattr(a, "options_approved_level", 0) or 0),
            options_trading_level=int(getattr(a, "options_trading_level", 0) or 0),
            pattern_day_trader=bool(a.pattern_day_trader),
            status=str(a.status),
        )

    def market_clock(self) -> dict:
        c = self.trading.get_clock()
        return {
            "is_open": bool(c.is_open),
            "timestamp": c.timestamp,
            "next_open": c.next_open,
            "next_close": c.next_close,
        }

    def positions(self) -> pd.Series:
        """
        Current equity positions as a Series of market values, indexed by symbol.

        Option positions are excluded here on purpose: they enter the risk
        calculation through their Greeks, not as a linear market value, and
        mixing the two would double-count exposure.
        """
        rows: dict[str, float] = {}
        for p in self.trading.get_all_positions():
            if _is_option(p):
                continue
            rows[p.symbol] = float(p.market_value or 0)
        return pd.Series(rows, dtype=float)

    def option_positions(self) -> list[dict]:
        """Open option positions, kept separate from the equity book."""
        out = []
        for p in self.trading.get_all_positions():
            if _is_option(p):
                out.append({
                    "symbol": p.symbol,
                    "qty": float(p.qty or 0),
                    "market_value": float(p.market_value or 0),
                    "avg_entry_price": float(p.avg_entry_price or 0),
                    "unrealized_pl": float(p.unrealized_pl or 0),
                })
        return out

    # -- market data --------------------------------------------------------

    def daily_closes(
        self,
        symbols: list[str],
        lookback_days: int = 400,
        end: dt.datetime | None = None,
    ) -> pd.DataFrame:
        """
        Daily close prices as a (dates x symbols) panel — the input to the
        covariance matrix.

        `lookback_days` is CALENDAR days, so ask for ~400 to reliably land 252
        trading days. Requesting 252 calendar days quietly gives you ~172
        observations and a covariance matrix estimated on too little data.
        """
        end = end or dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=20)
        start = end - dt.timedelta(days=lookback_days)

        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=self.feed,
        )
        bars = self.stocks.get_stock_bars(req)
        df = bars.df
        if df.empty:
            raise RuntimeError(
                f"No bars returned for {symbols}. Check the symbols are valid and "
                f"that your data feed ({self.feed.value}) covers them."
            )

        # alpaca-py returns a MultiIndex (symbol, timestamp); pivot to a panel.
        panel = df["close"].unstack(level=0)
        panel.index = pd.to_datetime(panel.index).tz_localize(None).normalize()
        return panel.sort_index().dropna(how="all")

    def latest_news(
        self,
        symbols: list[str],
        hours_back: int = 24,
        limit: int = 50,
    ) -> list[dict]:
        """
        Recent headlines for the universe — the Sentiment Agent's input.

        Returns plain dicts rather than SDK objects so the agent layer never
        depends on the SDK's model classes.
        """
        start = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours_back)
        req = NewsRequest(
            symbols=",".join(symbols),
            start=start,
            limit=limit,
            include_content=False,
            exclude_contentless=True,
        )
        result = self.news_client.get_news(req)
        items = result.data.get("news", []) if hasattr(result, "data") else []

        return [
            {
                "headline": getattr(n, "headline", ""),
                "summary": getattr(n, "summary", ""),
                "symbols": list(getattr(n, "symbols", []) or []),
                "source": getattr(n, "source", ""),
                "created_at": str(getattr(n, "created_at", "")),
                "url": getattr(n, "url", ""),
            }
            for n in items
        ]

    # -- options ------------------------------------------------------------

    def option_contracts(
        self,
        underlying: str,
        min_dte: int = 7,
        max_dte: int = 21,
        contract_type: ContractType = ContractType.PUT,
        strike_low: float | None = None,
        strike_high: float | None = None,
        limit: int = 200,
    ) -> list[dict]:
        """
        Tradable contracts for one underlying within a days-to-expiry window.

        DTE is converted to explicit expiry dates because the API filters on
        dates, not on tenor.
        """
        today = dt.date.today()
        req = GetOptionContractsRequest(
            underlying_symbols=[underlying],
            status=AssetStatus.ACTIVE,
            expiration_date_gte=today + dt.timedelta(days=min_dte),
            expiration_date_lte=today + dt.timedelta(days=max_dte),
            type=contract_type,
            strike_price_gte=str(strike_low) if strike_low else None,
            strike_price_lte=str(strike_high) if strike_high else None,
            limit=limit,
        )
        res = self.trading.get_option_contracts(req)
        contracts = getattr(res, "option_contracts", []) or []

        return [
            {
                "symbol": c.symbol,
                "underlying": c.underlying_symbol,
                "type": str(c.type.value if hasattr(c.type, "value") else c.type),
                "strike": float(c.strike_price),
                "expiration": str(c.expiration_date),
                "dte": (c.expiration_date - today).days,
                "open_interest": int(c.open_interest or 0)
                if getattr(c, "open_interest", None)
                else 0,
            }
            for c in contracts
        ]

    def option_snapshots(self, symbols: list[str]) -> dict[str, dict]:
        """
        Implied volatility and Greeks per contract — the pricing input for both
        the edge signal and strike selection.

        Contracts with no IV are dropped rather than defaulted: a missing IV
        means the contract is illiquid, and an illiquid hedge is not a hedge.
        """
        if not symbols:
            return {}

        req = OptionSnapshotRequest(
            symbol_or_symbols=symbols, feed=self.options_feed
        )
        snaps = self.options.get_option_snapshot(req)

        out: dict[str, dict] = {}
        for sym, s in snaps.items():
            iv = getattr(s, "implied_volatility", None)
            if iv is None:
                continue
            g = getattr(s, "greeks", None)
            q = getattr(s, "latest_quote", None)
            out[sym] = {
                "implied_volatility": float(iv),
                "delta": float(getattr(g, "delta", 0.0) or 0.0) if g else None,
                "gamma": float(getattr(g, "gamma", 0.0) or 0.0) if g else None,
                "theta": float(getattr(g, "theta", 0.0) or 0.0) if g else None,
                "vega": float(getattr(g, "vega", 0.0) or 0.0) if g else None,
                "bid": float(getattr(q, "bid_price", 0.0) or 0.0) if q else None,
                "ask": float(getattr(q, "ask_price", 0.0) or 0.0) if q else None,
            }
        return out
