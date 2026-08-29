"""
Shield-AI — formal statistical validation of the VaR model.

This is the module that turns "I used SciPy" into "my risk model is statistically
validated". Almost nobody in a hackathon backtests their risk model; a VaR number
nobody has tested is an assertion, not a measurement.

The idea
--------
A 99% VaR makes a falsifiable prediction: the realised loss should exceed the VaR
estimate on about 1% of days, and those exceedances should be scattered randomly
in time rather than arriving in clusters. Two classical likelihood-ratio tests
check exactly those two claims.

  * Kupiec (1995) — Proportion of Failures. Is the exceedance RATE correct?
  * Christoffersen (1998) — Independence. Are exceedances INDEPENDENT, or does
    one breach predict the next? A model that fails this is dangerous even with
    a perfect hit rate: it means the model stays broken for days at a time,
    which is precisely when losses compound.

Both statistics are asymptotically chi-squared, so SciPy gives us p-values
directly. A p-value below 0.05 rejects the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class BacktestResult:
    """Outcome of a VaR backtest over a series of days."""

    n_observations: int
    n_exceptions: int
    expected_exceptions: float
    confidence: float

    kupiec_stat: float
    kupiec_pvalue: float

    christoffersen_ind_stat: float
    christoffersen_ind_pvalue: float

    conditional_coverage_stat: float
    conditional_coverage_pvalue: float

    @property
    def exception_rate(self) -> float:
        return self.n_exceptions / self.n_observations if self.n_observations else 0.0

    @property
    def passes(self) -> bool:
        """Model is accepted if neither test rejects at the 5% level."""
        return self.kupiec_pvalue > 0.05 and self.christoffersen_ind_pvalue > 0.05

    def summary(self) -> str:
        verdict = "ACCEPTED" if self.passes else "REJECTED"
        return (
            f"VaR model {verdict} at {self.confidence:.0%} confidence\n"
            f"  Observations        : {self.n_observations}\n"
            f"  Exceptions          : {self.n_exceptions} "
            f"(expected {self.expected_exceptions:.1f})\n"
            f"  Exception rate      : {self.exception_rate:.2%} "
            f"(target {1 - self.confidence:.2%})\n"
            f"  Kupiec POF          : LR={self.kupiec_stat:.3f}  "
            f"p={self.kupiec_pvalue:.4f}\n"
            f"  Christoffersen IND  : LR={self.christoffersen_ind_stat:.3f}  "
            f"p={self.christoffersen_ind_pvalue:.4f}\n"
            f"  Conditional coverage: LR={self.conditional_coverage_stat:.3f}  "
            f"p={self.conditional_coverage_pvalue:.4f}"
        )


def _kupiec_pof(n: int, x: int, p: float) -> tuple[float, float]:
    """
    Kupiec Proportion-of-Failures likelihood-ratio test.

        LR_POF = -2 * ln[ L(p) / L(p_hat) ]

    where L(p) is the likelihood of seeing x exceptions in n days if the true
    exceedance probability is the model's claimed p, and L(p_hat) uses the
    observed rate x/n. Under the null that the model is correct, LR ~ chi2(1).

    Intuition: if the model claims 1% and you observed 1%, the two likelihoods
    are nearly equal, the ratio is near 1, its log is near 0, and the statistic
    is small — the model survives.
    """
    if x == 0:
        # Closed form when no exceptions occurred; the general formula has a
        # 0*log(0) term that must be evaluated as its limit, 0.
        lr = -2.0 * n * np.log(1.0 - p)
    elif x == n:
        lr = -2.0 * n * np.log(p)
    else:
        p_hat = x / n
        log_l_null = (n - x) * np.log(1.0 - p) + x * np.log(p)
        log_l_alt = (n - x) * np.log(1.0 - p_hat) + x * np.log(p_hat)
        lr = -2.0 * (log_l_null - log_l_alt)

    return float(lr), float(1.0 - stats.chi2.cdf(lr, df=1))


def _christoffersen_independence(exceptions: np.ndarray) -> tuple[float, float]:
    """
    Christoffersen independence test.

    Treats the exception sequence as a two-state Markov chain and asks whether
    the probability of an exception tomorrow depends on whether there was one
    today. Counts the four transition types:

        n00: no exception  -> no exception
        n01: no exception  -> exception
        n10: exception     -> no exception
        n11: exception     -> exception

    Under independence, pi01 == pi11 == pi. If breaches cluster, pi11 >> pi01
    and the statistic blows up. LR ~ chi2(1).
    """
    e = np.asarray(exceptions).astype(int)
    if e.size < 2:
        return 0.0, 1.0

    prev, curr = e[:-1], e[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    # Transition probabilities. If a row is empty the chain never visited that
    # state, and that row contributes nothing to the likelihood.
    denom0, denom1 = n00 + n01, n10 + n11
    if denom0 == 0 or denom1 == 0:
        return 0.0, 1.0

    pi01 = n01 / denom0
    pi11 = n11 / denom1
    pi = (n01 + n11) / (denom0 + denom1)

    if pi in (0.0, 1.0) or pi01 in (0.0,) and pi11 in (0.0,):
        return 0.0, 1.0

    def _safe_log(value: float) -> float:
        return np.log(value) if value > 0 else 0.0

    log_l_null = (n00 + n10) * _safe_log(1 - pi) + (n01 + n11) * _safe_log(pi)
    log_l_alt = (
        n00 * _safe_log(1 - pi01)
        + n01 * _safe_log(pi01)
        + n10 * _safe_log(1 - pi11)
        + n11 * _safe_log(pi11)
    )

    lr = -2.0 * (log_l_null - log_l_alt)
    lr = max(lr, 0.0)  # guard against tiny negative values from floating point
    return float(lr), float(1.0 - stats.chi2.cdf(lr, df=1))


def backtest_var(
    realised_returns: pd.Series | np.ndarray,
    var_estimates: pd.Series | np.ndarray,
    confidence: float = 0.99,
) -> BacktestResult:
    """
    Run the full VaR backtest.

    Parameters
    ----------
    realised_returns
        Actual portfolio returns, one per day (negative = loss).
    var_estimates
        The VaR forecast made for that same day, as a POSITIVE loss fraction —
        matching the convention in risk.py. Must be aligned and equal length.

    An "exception" is a day where the realised loss exceeded the VaR estimate,
    i.e. realised_return < -var_estimate.
    """
    r = np.asarray(realised_returns, dtype=float)
    v = np.asarray(var_estimates, dtype=float)

    if r.shape != v.shape:
        raise ValueError(
            f"Length mismatch: {r.shape[0]} returns vs {v.shape[0]} VaR estimates. "
            "Each VaR forecast must be paired with the day it was forecasting."
        )

    exceptions = (r < -v).astype(int)
    n = int(r.size)
    x = int(exceptions.sum())
    p = 1.0 - confidence

    kupiec_stat, kupiec_p = _kupiec_pof(n, x, p)
    ind_stat, ind_p = _christoffersen_independence(exceptions)

    # Conditional coverage is the joint test: correct rate AND independent.
    cc_stat = kupiec_stat + ind_stat
    cc_p = float(1.0 - stats.chi2.cdf(cc_stat, df=2))

    return BacktestResult(
        n_observations=n,
        n_exceptions=x,
        expected_exceptions=n * p,
        confidence=confidence,
        kupiec_stat=kupiec_stat,
        kupiec_pvalue=kupiec_p,
        christoffersen_ind_stat=ind_stat,
        christoffersen_ind_pvalue=ind_p,
        conditional_coverage_stat=cc_stat,
        conditional_coverage_pvalue=cc_p,
    )
