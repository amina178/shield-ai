"""
Shield-AI — deterministic portfolio risk engine.

This module contains no AI and no opinions. It turns a price history and a set
of positions into hard numbers: Value-at-Risk, Conditional VaR, and
beta-weighted net Delta. The cognitive layer is only ever allowed to read these
numbers, never to compute or override them.

Sign convention used everywhere in this file
--------------------------------------------
VaR and CVaR are returned as POSITIVE numbers representing a LOSS.
A 99% VaR of 0.023 means: "on 99 days out of 100 the one-day loss should not
exceed 2.3% of portfolio value." This is the convention risk managers expect,
and getting it backwards is the single most common bug in student VaR code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a price panel (rows = dates, columns = symbols) into log returns.

    Log returns rather than simple returns because they are additive across
    time, which is what makes the sqrt-of-time scaling below legitimate.
    """
    if prices.isna().any().any():
        prices = prices.ffill().dropna(how="any")
    return np.log(prices / prices.shift(1)).dropna(how="any")


def portfolio_weights(position_values: pd.Series) -> pd.Series:
    """
    Normalise position market values into weights that sum to 1.

    Note these are *gross* weights over the invested notional, not over account
    equity — cash is handled separately, because cash has no return variance
    and would otherwise dilute the covariance estimate.
    """
    total = position_values.abs().sum()
    if total == 0:
        raise ValueError("Cannot compute weights for an empty portfolio.")
    return position_values / total


# ---------------------------------------------------------------------------
# Value-at-Risk — three independent estimators
# ---------------------------------------------------------------------------
# We compute VaR three ways on purpose. If the three disagree materially, the
# distributional assumption is wrong and that is itself a risk signal. Reporting
# only one number hides that information.

def parametric_var(
    returns: pd.DataFrame,
    weights: pd.Series,
    confidence: float = 0.99,
    horizon_days: int = 1,
    student_t_df: float | None = None,
) -> float:
    """
    Variance-covariance (Gaussian or Student-t) VaR.

    Fast and analytic: it only needs the mean vector and covariance matrix.
    Its weakness is that it assumes elliptical returns, which understates
    crash risk — hence the optional Student-t override.
    """
    w = weights.reindex(returns.columns).fillna(0.0).to_numpy()
    mu = returns.mean().to_numpy() @ w
    cov = returns.cov().to_numpy()
    sigma = float(np.sqrt(w @ cov @ w))

    if student_t_df is not None:
        # Scale so the t-distribution has unit variance before applying sigma,
        # otherwise the heavy tails silently inflate the whole distribution.
        df = student_t_df
        q = stats.t.ppf(1.0 - confidence, df) * np.sqrt((df - 2.0) / df)
    else:
        q = stats.norm.ppf(1.0 - confidence)

    # Square-root-of-time scaling. Valid for i.i.d. returns; GARCH gives us a
    # better multi-day answer, which is why the horizon defaults to 1 day.
    scale = np.sqrt(horizon_days)
    var_fraction = -(mu * horizon_days + q * sigma * scale)
    return float(max(var_fraction, 0.0))


def historical_var(
    returns: pd.DataFrame,
    weights: pd.Series,
    confidence: float = 0.99,
) -> float:
    """
    Historical simulation VaR — the empirical quantile of realised portfolio
    returns. Assumes nothing about the distribution, but can only ever show
    losses that already happened in the lookback window.
    """
    w = weights.reindex(returns.columns).fillna(0.0).to_numpy()
    port = returns.to_numpy() @ w
    var_fraction = -np.percentile(port, (1.0 - confidence) * 100.0)
    return float(max(var_fraction, 0.0))


def monte_carlo_var(
    returns: pd.DataFrame,
    weights: pd.Series,
    confidence: float = 0.99,
    paths: int = 50_000,
    student_t_df: float | None = 5.0,
    seed: int | None = 42,
) -> float:
    """
    Monte Carlo VaR under a multivariate Student-t.

    Draws correlated shocks from the estimated covariance matrix. Unlike
    historical simulation it can generate losses worse than anything observed,
    which is the entire point of a tail model.
    """
    rng = np.random.default_rng(seed)
    w = weights.reindex(returns.columns).fillna(0.0).to_numpy()
    mu = returns.mean().to_numpy()
    cov = returns.cov().to_numpy()

    # Cholesky needs positive definiteness; nudge the diagonal if the estimate
    # is near-singular (happens with highly correlated names or short history).
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        chol = np.linalg.cholesky(cov + np.eye(len(cov)) * 1e-10)

    n = len(mu)
    if student_t_df is not None:
        df = student_t_df
        z = rng.standard_normal((paths, n))
        # Multivariate t = Gaussian / sqrt(chi2/df). Rescale to unit variance.
        chi = rng.chisquare(df, size=(paths, 1)) / df
        shocks = z / np.sqrt(chi) * np.sqrt((df - 2.0) / df)
    else:
        shocks = rng.standard_normal((paths, n))

    sims = mu + shocks @ chol.T
    port = sims @ w
    var_fraction = -np.percentile(port, (1.0 - confidence) * 100.0)
    return float(max(var_fraction, 0.0))


def conditional_var(
    returns: pd.DataFrame,
    weights: pd.Series,
    confidence: float = 0.99,
) -> float:
    """
    Conditional VaR (Expected Shortfall): the average loss *given* that the VaR
    threshold was breached.

    CVaR answers the question VaR refuses to: "and how bad is it when it does
    go wrong?" It is also coherent as a risk measure, which VaR is not.
    """
    w = weights.reindex(returns.columns).fillna(0.0).to_numpy()
    port = returns.to_numpy() @ w
    threshold = np.percentile(port, (1.0 - confidence) * 100.0)
    tail = port[port <= threshold]
    if tail.size == 0:
        return 0.0
    return float(max(-tail.mean(), 0.0))


# ---------------------------------------------------------------------------
# Directional exposure
# ---------------------------------------------------------------------------

def beta_weighted_delta(
    returns: pd.DataFrame,
    position_values: pd.Series,
    equity: float,
    benchmark: str = "SPY",
) -> float:
    """
    Net portfolio delta expressed in benchmark-equivalent units, divided by
    account equity.

    Why beta-weight at all: $10,000 of NVDA and $10,000 of a utility are not the
    same market exposure. Summing raw dollar deltas across names with different
    betas produces a number that means nothing. Beta-weighting converts every
    position into "how much SPY is this equivalent to", which is the only form
    in which deltas can legitimately be added together.

    Betas come from OLS of each asset's returns on the benchmark's returns.
    """
    if benchmark not in returns.columns:
        raise ValueError(f"Benchmark {benchmark!r} missing from returns panel.")
    if equity <= 0:
        raise ValueError("Equity must be positive.")

    bench = returns[benchmark].to_numpy()
    bench_var = bench.var(ddof=1)

    exposure = 0.0
    for symbol, value in position_values.items():
        if symbol == benchmark:
            beta = 1.0
        elif symbol in returns.columns:
            asset = returns[symbol].to_numpy()
            # beta = Cov(asset, bench) / Var(bench) — the OLS slope.
            beta = float(np.cov(asset, bench, ddof=1)[0, 1] / bench_var)
        else:
            beta = 1.0  # conservative default for an unmodelled name
        exposure += float(value) * beta

    return float(exposure / equity)


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskReport:
    """Everything the guardrail check and the dashboard need, in one object."""

    # All VaR/CVaR figures below are fractions of ACCOUNT EQUITY, already
    # adjusted for how much of the account is actually invested.
    equity: float
    invested: float
    var_parametric: float
    var_historical: float
    var_monte_carlo: float
    cvar: float
    net_delta: float
    n_observations: int

    @property
    def gross_leverage(self) -> float:
        """Invested notional divided by equity. Below 1.0 means cash on hand."""
        return self.invested / self.equity if self.equity else 0.0

    @property
    def var_headline(self) -> float:
        """
        The number the mandate is enforced against.

        We take the maximum of the three estimators. This is deliberately
        conservative: if any reasonable model says the risk is large, the agent
        treats the risk as large. Averaging would let an optimistic model dilute
        a pessimistic one, which is exactly backwards for a risk system.
        """
        return max(self.var_parametric, self.var_historical, self.var_monte_carlo)

    @property
    def var_dollars(self) -> float:
        return self.var_headline * self.equity

    @property
    def model_disagreement(self) -> float:
        """
        Spread between the most and least conservative estimator, as a fraction
        of the headline. A large spread means the distributional assumption is
        doing heavy lifting and the number should be trusted less.
        """
        lo = min(self.var_parametric, self.var_historical, self.var_monte_carlo)
        return (self.var_headline - lo) / self.var_headline if self.var_headline else 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "equity": self.equity,
            "invested": self.invested,
            "var_parametric": self.var_parametric,
            "var_historical": self.var_historical,
            "var_monte_carlo": self.var_monte_carlo,
            "var_headline": self.var_headline,
            "var_dollars": self.var_dollars,
            "cvar": self.cvar,
            "net_delta": self.net_delta,
            "model_disagreement": self.model_disagreement,
            "n_observations": self.n_observations,
        }


def build_risk_report(
    prices: pd.DataFrame,
    position_values: pd.Series,
    equity: float,
    confidence: float = 0.99,
    student_t_df: float = 5.0,
    mc_paths: int = 50_000,
    benchmark: str = "SPY",
    min_observations: int = 60,
) -> RiskReport:
    """Run the full deterministic risk pass over the current book."""
    rets = log_returns(prices)
    if len(rets) < min_observations:
        raise ValueError(
            f"Only {len(rets)} return observations; need {min_observations}. "
            "Refusing to estimate risk on insufficient history."
        )

    weights = portfolio_weights(position_values)
    invested = float(position_values.abs().sum())

    # ------------------------------------------------------------------
    # UNITS. The estimators above take NORMALISED weights (summing to 1), so
    # they return VaR as a fraction of INVESTED NOTIONAL. The mandate, the
    # dashboard and every dollar figure are expressed as a fraction of ACCOUNT
    # EQUITY. Those two are only the same number when the book is 100% invested.
    #
    # Converting requires gross leverage = invested / equity. Skipping this step
    # is a silent and dangerous bug: a 55%-invested book would report ~1.8x its
    # true risk, the agent would hedge positions that were never in breach, and
    # — worse — scaling the book down would not reduce the reported VaR at all,
    # because proportional scaling leaves normalised weights unchanged.
    # ------------------------------------------------------------------
    gross_leverage = invested / float(equity)

    return RiskReport(
        equity=float(equity),
        invested=invested,
        var_parametric=parametric_var(
            rets, weights, confidence, student_t_df=student_t_df
        ) * gross_leverage,
        var_historical=historical_var(rets, weights, confidence) * gross_leverage,
        var_monte_carlo=monte_carlo_var(
            rets, weights, confidence, paths=mc_paths, student_t_df=student_t_df
        ) * gross_leverage,
        cvar=conditional_var(rets, weights, confidence) * gross_leverage,
        net_delta=beta_weighted_delta(rets, position_values, equity, benchmark),
        n_observations=len(rets),
    )
