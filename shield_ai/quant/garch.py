"""
Shield-AI — GARCH(1,1) volatility forecasting.

This module does double duty, which is the central design idea of Shield-AI:

  1. RISK.  The conditional volatility forecast feeds the VaR engine, so the
     risk budget tightens automatically when the market gets rough.

  2. ALPHA. The same forecast is compared against option-implied volatility.
     The gap between them — the variance risk premium — is the agent's entry
     signal. IV materially above the forecast means options are expensive
     relative to what the model expects to actually happen: sell premium.
     IV materially below means protection is cheap: buy it.

One model, two jobs. That is why the risk engine and the trading strategy are
not two bolted-together systems here.

Why GARCH(1,1) specifically
---------------------------
Volatility clusters: violent days follow violent days. A plain rolling standard
deviation treats a shock from 200 days ago exactly like yesterday's, so it
reacts late and decays late. GARCH(1,1) models today's variance as a weighted
blend of a long-run average, yesterday's surprise, and yesterday's variance:

    sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2

alpha controls how sharply it reacts to news, beta how long the memory lasts.
alpha + beta is the persistence: close to 1 means shocks fade slowly. For daily
equity returns it is typically ~0.95-0.99, which is exactly why a calm rolling
window is dangerously optimistic on the day after a shock.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model

TRADING_DAYS = 252


@dataclass(frozen=True)
class VolForecast:
    """A volatility forecast with the diagnostics needed to trust or reject it."""

    symbol: str
    annualized_vol: float      # forecast, as a decimal (0.24 == 24%)
    daily_vol: float
    horizon_days: int
    omega: float
    alpha: float
    beta: float
    converged: bool

    @property
    def persistence(self) -> float:
        """alpha + beta. Above 1.0 means the model is non-stationary — reject it."""
        return self.alpha + self.beta

    @property
    def is_trustworthy(self) -> bool:
        return self.converged and 0.0 < self.persistence < 1.0

    @property
    def vol_points(self) -> float:
        """Annualised vol expressed in percentage points, e.g. 24.3."""
        return self.annualized_vol * 100.0


def forecast_volatility(
    returns: pd.Series,
    symbol: str = "",
    horizon_days: int = 30,
    rescale_factor: float = 100.0,
) -> VolForecast:
    """
    Fit GARCH(1,1) to a daily return series and forecast average volatility
    over the next `horizon_days`.

    `horizon_days` should match the option expiry you are pricing against —
    comparing a 1-day forecast to a 30-day implied vol is an apples-to-oranges
    error that will make every option look expensive.

    Returns are multiplied by `rescale_factor` before fitting purely for
    numerical conditioning: daily returns are ~0.01, and the optimiser behaves
    far better on numbers near 1. The scaling is undone before returning.
    """
    r = returns.dropna()
    if len(r) < 100:
        raise ValueError(
            f"GARCH needs at least 100 observations, got {len(r)}. "
            "Fitting on less produces unstable parameters."
        )

    scaled = r * rescale_factor

    model = arch_model(scaled, vol="GARCH", p=1, q=1, mean="Constant", dist="t")
    res = model.fit(disp="off", show_warning=False)

    fc = res.forecast(horizon=horizon_days, reindex=False)

    # fc.variance holds the per-day forecast variance path. The volatility of
    # an average over the horizon is the sqrt of the MEAN of the daily
    # variances — not the mean of the daily vols. Averaging vols instead of
    # variances is a classic and quietly wrong shortcut.
    daily_var_path = fc.variance.to_numpy()[-1]
    mean_daily_var = float(np.mean(daily_var_path)) / (rescale_factor ** 2)
    daily_vol = float(np.sqrt(mean_daily_var))

    params = res.params
    return VolForecast(
        symbol=symbol,
        annualized_vol=daily_vol * np.sqrt(TRADING_DAYS),
        daily_vol=daily_vol,
        horizon_days=horizon_days,
        omega=float(params.get("omega", np.nan)),
        alpha=float(params.get("alpha[1]", np.nan)),
        beta=float(params.get("beta[1]", np.nan)),
        converged=bool(res.convergence_flag == 0),
    )


# ---------------------------------------------------------------------------
# The trading signal
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VolEdge:
    """
    The variance risk premium for one underlying, in volatility points.

    Positive edge  -> options are rich vs. the model  -> sell premium (income)
    Negative edge  -> options are cheap vs. the model -> buy protection
    """

    symbol: str
    implied_vol_points: float
    forecast_vol_points: float
    horizon_days: int

    @property
    def edge_points(self) -> float:
        return self.implied_vol_points - self.forecast_vol_points

    @property
    def edge_ratio(self) -> float:
        """IV / forecast. Scale-free, so it compares across low- and high-vol names."""
        if self.forecast_vol_points <= 0:
            return float("nan")
        return self.implied_vol_points / self.forecast_vol_points

    def signal(self, min_edge_points: float = 4.0) -> str:
        """Discretise the edge into an action the policy layer can act on."""
        if self.edge_points >= min_edge_points:
            return "SELL_PREMIUM"
        if self.edge_points <= -min_edge_points:
            return "BUY_PROTECTION"
        return "HOLD"


def compute_edge(
    implied_vol: float,
    forecast: VolForecast,
) -> VolEdge:
    """
    Compare option-implied volatility against the GARCH forecast.

    `implied_vol` is a decimal (0.28 for 28%), as returned by Alpaca's
    get_option_snapshot. Both sides must be annualised and refer to the same
    horizon, which is why the forecast carries its horizon with it.
    """
    return VolEdge(
        symbol=forecast.symbol,
        implied_vol_points=implied_vol * 100.0,
        forecast_vol_points=forecast.vol_points,
        horizon_days=forecast.horizon_days,
    )
