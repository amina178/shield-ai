"""
Shield-AI — verification of the quantitative core.

These are not smoke tests. Each one constructs synthetic data whose correct
answer is known analytically, then checks the code reproduces it. That is the
difference between "my code runs" and "my code is right" — and it is the
evidence you show a judge who asks how you know your VaR is not nonsense.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shield_ai.quant.risk import (
    parametric_var,
    historical_var,
    monte_carlo_var,
    conditional_var,
    beta_weighted_delta,
    log_returns,
    build_risk_report,
)
from shield_ai.quant.garch import forecast_volatility, compute_edge
from shield_ai.quant.var_backtest import backtest_var

RNG = np.random.default_rng(20260829)


def _gaussian_panel(n_days=2000, sigma=0.01, n_assets=1, names=None):
    """Independent Gaussian returns with known sigma, as a price panel."""
    names = names or [f"A{i}" for i in range(n_assets)]
    rets = RNG.normal(0.0, sigma, size=(n_days, len(names)))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, columns=names), pd.DataFrame(rets, columns=names)


# ---------------------------------------------------------------------------
# 1. Parametric VaR must reproduce the analytic Gaussian quantile
# ---------------------------------------------------------------------------

def test_parametric_var_matches_analytic_gaussian():
    sigma = 0.01
    _, rets = _gaussian_panel(n_days=50_000, sigma=sigma, names=["A"])
    w = pd.Series({"A": 1.0})

    got = parametric_var(rets, w, confidence=0.99, student_t_df=None)
    expected = 2.32635 * sigma  # -norm.ppf(0.01) * sigma, mu = 0

    assert abs(got - expected) < 0.0002, f"got {got:.6f}, expected {expected:.6f}"
    print(f"  parametric VaR   = {got:.5f}   analytic = {expected:.5f}  ✓")


def test_student_t_var_is_more_conservative_than_gaussian():
    """Fat tails must produce a LARGER loss estimate, never a smaller one."""
    _, rets = _gaussian_panel(n_days=5000, sigma=0.01, names=["A"])
    w = pd.Series({"A": 1.0})

    gaussian = parametric_var(rets, w, 0.99, student_t_df=None)
    student = parametric_var(rets, w, 0.99, student_t_df=5.0)

    assert student > gaussian, "Student-t VaR must exceed Gaussian VaR"
    print(f"  gaussian VaR     = {gaussian:.5f}")
    print(f"  student-t VaR    = {student:.5f}   (more conservative)  ✓")


# ---------------------------------------------------------------------------
# 2. The three estimators must agree on Gaussian data
# ---------------------------------------------------------------------------

def test_three_estimators_agree_on_gaussian_data():
    """
    On genuinely Gaussian data all three methods estimate the same quantity,
    so they must converge. If they disagree here, one of them is wrong.
    """
    _, rets = _gaussian_panel(n_days=20_000, sigma=0.01, names=["A"])
    w = pd.Series({"A": 1.0})

    par = parametric_var(rets, w, 0.99, student_t_df=None)
    his = historical_var(rets, w, 0.99)
    mc = monte_carlo_var(rets, w, 0.99, paths=200_000, student_t_df=None)

    spread = max(par, his, mc) - min(par, his, mc)
    assert spread / par < 0.05, f"estimators disagree by {spread / par:.1%}"
    print(f"  parametric={par:.5f}  historical={his:.5f}  monte-carlo={mc:.5f}")
    print(f"  max spread       = {spread / par:.2%} of headline  ✓")


# ---------------------------------------------------------------------------
# 3. CVaR must always exceed VaR
# ---------------------------------------------------------------------------

def test_cvar_exceeds_var():
    """
    Expected Shortfall is the mean of the tail BEYOND the VaR quantile, so it is
    mathematically impossible for it to be smaller. This catches sign errors.
    """
    _, rets = _gaussian_panel(n_days=5000, sigma=0.012, names=["A"])
    w = pd.Series({"A": 1.0})

    var = historical_var(rets, w, 0.99)
    cvar = conditional_var(rets, w, 0.99)

    assert cvar > var, f"CVaR {cvar:.5f} must exceed VaR {var:.5f}"
    print(f"  VaR  = {var:.5f}")
    print(f"  CVaR = {cvar:.5f}   ratio = {cvar / var:.2f}x  ✓")


# ---------------------------------------------------------------------------
# 4. Beta-weighted delta must recover a beta we construct by hand
# ---------------------------------------------------------------------------

def test_beta_weighted_delta_recovers_known_beta():
    """
    Build an asset that is exactly 2x the benchmark plus noise. OLS must recover
    beta = 2, so $10k of it inside $100k equity must read as 0.20 net delta.
    """
    n = 5000
    bench = RNG.normal(0.0, 0.01, n)
    asset = 2.0 * bench + RNG.normal(0.0, 0.0005, n)  # tiny idiosyncratic noise

    rets = pd.DataFrame({"SPY": bench, "LEVER": asset})
    positions = pd.Series({"LEVER": 10_000.0})

    delta = beta_weighted_delta(rets, positions, equity=100_000.0, benchmark="SPY")

    assert abs(delta - 0.20) < 0.005, f"expected 0.20, got {delta:.4f}"
    print(f"  net delta        = {delta:.4f}   expected 0.2000  ✓")


def test_benchmark_position_has_unit_beta():
    """A position in the benchmark itself must contribute delta 1:1."""
    n = 2000
    bench = RNG.normal(0.0, 0.01, n)
    rets = pd.DataFrame({"SPY": bench, "OTHER": RNG.normal(0.0, 0.01, n)})
    positions = pd.Series({"SPY": 50_000.0})

    delta = beta_weighted_delta(rets, positions, equity=100_000.0, benchmark="SPY")
    assert abs(delta - 0.50) < 1e-9
    print(f"  SPY-only delta   = {delta:.4f}   expected 0.5000  ✓")


# ---------------------------------------------------------------------------
# 5. Kupiec must ACCEPT a correct model and REJECT a broken one
# ---------------------------------------------------------------------------

def test_kupiec_accepts_correctly_calibrated_var():
    """
    Feed the backtest a VaR that is genuinely the 99% quantile of the return
    distribution. Exceptions should land near 1% and the test should not reject.
    """
    n = 2000
    sigma = 0.01
    rets = RNG.normal(0.0, sigma, n)
    var_series = np.full(n, 2.32635 * sigma)  # the true 99% quantile

    res = backtest_var(rets, var_series, confidence=0.99)

    assert res.kupiec_pvalue > 0.05, f"wrongly rejected, p={res.kupiec_pvalue:.4f}"
    print(f"  exceptions       = {res.n_exceptions} of {res.n_observations} "
          f"(expected {res.expected_exceptions:.0f})")
    print(f"  Kupiec p-value   = {res.kupiec_pvalue:.4f}  -> ACCEPT  ✓")


def test_kupiec_rejects_an_overconfident_var():
    """
    A model that claims 99% but is really sized for ~90% will breach far too
    often. The test must catch it — otherwise it has no power and is decoration.
    """
    n = 2000
    sigma = 0.01
    rets = RNG.normal(0.0, sigma, n)
    var_series = np.full(n, 1.28155 * sigma)  # only the 90% quantile

    res = backtest_var(rets, var_series, confidence=0.99)

    assert res.kupiec_pvalue < 0.01, f"failed to reject, p={res.kupiec_pvalue:.4f}"
    print(f"  exceptions       = {res.n_exceptions} of {res.n_observations} "
          f"(expected {res.expected_exceptions:.0f})")
    print(f"  Kupiec p-value   = {res.kupiec_pvalue:.2e}  -> REJECT  ✓")


# ---------------------------------------------------------------------------
# 6. Christoffersen must detect clustered breaches
# ---------------------------------------------------------------------------

def test_christoffersen_detects_clustering():
    """
    Construct exceptions that arrive in tight blocks — the signature of a model
    that stays broken for days. The hit RATE is correct, so Kupiec alone would
    pass it. Only the independence test can catch this failure mode.
    """
    n = 1000
    sigma = 0.01
    # The background must be quiet enough that NO exception happens by chance,
    # otherwise random breaches dilute the clustering we are trying to detect.
    # At 0.1x sigma against a threshold built on 0.3x sigma, a natural breach
    # would need a ~7-sigma move: effectively impossible.
    rets = RNG.normal(0.0, sigma * 0.1, n)
    var_series = np.full(n, 2.32635 * sigma * 0.3)

    # Ten exceptions, arranged as two clusters of five consecutive days.
    for start in (200, 600):
        for k in range(5):
            rets[start + k] = -var_series[0] * 1.5

    res = backtest_var(rets, var_series, confidence=0.99)

    assert res.n_exceptions == 10, res.n_exceptions
    assert res.kupiec_pvalue > 0.05, "rate is correct, Kupiec should pass"
    assert res.christoffersen_ind_pvalue < 0.05, "clustering not detected"
    print(f"  exceptions       = {res.n_exceptions} (correct rate)")
    print(f"  Kupiec p-value   = {res.kupiec_pvalue:.4f}  -> passes rate test")
    print(f"  Christoffersen p = {res.christoffersen_ind_pvalue:.2e}  -> "
          f"REJECT for clustering  ✓")


# ---------------------------------------------------------------------------
# 7. GARCH must recover parameters it was simulated from
# ---------------------------------------------------------------------------

def _simulate_garch(n, omega, alpha, beta, seed=7):
    """Simulate a GARCH(1,1) path on the x100 scale used by the fitter."""
    rng = np.random.default_rng(seed)
    var_lr = omega / (1 - alpha - beta)
    sigma2 = np.empty(n)
    eps = np.empty(n)
    sigma2[0] = var_lr
    eps[0] = rng.normal(0, np.sqrt(sigma2[0]))
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = rng.normal(0, np.sqrt(sigma2[t]))
    return eps / 100.0, var_lr  # return to native return scale


def test_garch_recovers_simulated_parameters():
    omega, alpha, beta = 0.05, 0.08, 0.90
    series, var_lr = _simulate_garch(4000, omega, alpha, beta)
    returns = pd.Series(series)

    fc = forecast_volatility(returns, symbol="SIM", horizon_days=30)

    true_annual_vol = np.sqrt(var_lr) / 100.0 * np.sqrt(252)

    assert fc.is_trustworthy, "model did not converge or is non-stationary"
    assert abs(fc.alpha - alpha) < 0.05, f"alpha {fc.alpha:.3f} vs {alpha}"
    assert abs(fc.beta - beta) < 0.06, f"beta {fc.beta:.3f} vs {beta}"
    assert abs(fc.annualized_vol - true_annual_vol) < 0.10

    print(f"  alpha  fitted={fc.alpha:.4f}  true={alpha}")
    print(f"  beta   fitted={fc.beta:.4f}  true={beta}")
    print(f"  persistence    = {fc.persistence:.4f}")
    print(f"  ann. vol       = {fc.annualized_vol:.2%}  "
          f"long-run truth = {true_annual_vol:.2%}  ✓")


# ---------------------------------------------------------------------------
# 8. The edge signal must fire in the right direction
# ---------------------------------------------------------------------------

def test_edge_signal_directions():
    series, _ = _simulate_garch(1500, 0.05, 0.08, 0.90, seed=11)
    fc = forecast_volatility(pd.Series(series), symbol="SIM", horizon_days=30)
    forecast_pts = fc.vol_points

    rich = compute_edge((forecast_pts + 8.0) / 100.0, fc)
    cheap = compute_edge((forecast_pts - 8.0) / 100.0, fc)
    fair = compute_edge((forecast_pts + 1.0) / 100.0, fc)

    assert rich.signal(4.0) == "SELL_PREMIUM"
    assert cheap.signal(4.0) == "BUY_PROTECTION"
    assert fair.signal(4.0) == "HOLD"

    print(f"  forecast vol     = {forecast_pts:.2f} pts")
    print(f"  IV +8 pts        -> {rich.signal(4.0)}  (edge {rich.edge_points:+.1f})")
    print(f"  IV -8 pts        -> {cheap.signal(4.0)}  (edge {cheap.edge_points:+.1f})")
    print(f"  IV +1 pt         -> {fair.signal(4.0)}  (edge {fair.edge_points:+.1f})  ✓")


# ---------------------------------------------------------------------------
# 9. End-to-end report
# ---------------------------------------------------------------------------

def test_full_risk_report():
    n = 800
    bench = RNG.normal(0.0004, 0.009, n)
    panel = pd.DataFrame({
        "SPY": bench,
        "AAPL": 1.15 * bench + RNG.normal(0, 0.008, n),
        "NVDA": 1.85 * bench + RNG.normal(0, 0.020, n),
    })
    prices = 100.0 * np.exp(panel.cumsum())

    positions = pd.Series({"SPY": 30_000.0, "AAPL": 15_000.0, "NVDA": 12_000.0})
    report = build_risk_report(prices, positions, equity=100_000.0, mc_paths=20_000)

    assert report.var_headline > 0
    assert report.cvar > 0
    assert 0 < report.net_delta < 2
    assert report.n_observations == n - 1

    print(f"  headline VaR     = {report.var_headline:.3%} "
          f"(${report.var_dollars:,.0f})")
    print(f"  CVaR             = {report.cvar:.3%}")
    print(f"  net delta        = {report.net_delta:.3f}")
    print(f"  model disagreement = {report.model_disagreement:.1%}")
    print(f"  observations     = {report.n_observations}  ✓")


# ---------------------------------------------------------------------------
# 10. Regression: VaR must be expressed in EQUITY terms, not invested notional
# ---------------------------------------------------------------------------

def test_var_scales_with_exposure():
    """
    Regression test for a real bug.

    The VaR estimators take NORMALISED weights, so they return risk as a
    fraction of invested notional. The mandate is a fraction of account equity.
    Without converting by gross leverage, a half-invested book reported the same
    VaR as a fully invested one — meaning the position-sizing loop could cut
    exposure forever without the reported risk ever falling.

    The invariant: halve the book, halve the VaR.
    """
    n = 600
    bench = RNG.normal(0.0003, 0.009, n)
    panel = pd.DataFrame({
        "SPY": bench,
        "AAPL": 1.10 * bench + RNG.normal(0, 0.008, n),
        "NVDA": 1.90 * bench + RNG.normal(0, 0.020, n),
    })
    prices = 100.0 * np.exp(panel.cumsum())
    equity = 100_000.0

    full = pd.Series({"SPY": 30_000.0, "AAPL": 15_000.0, "NVDA": 10_000.0})
    half = full * 0.5

    r_full = build_risk_report(prices, full, equity, mc_paths=10_000)
    r_half = build_risk_report(prices, half, equity, mc_paths=10_000)

    ratio = r_half.var_headline / r_full.var_headline
    assert 0.48 < ratio < 0.52, f"halving must halve VaR, got ratio {ratio:.3f}"

    # And the dollar figure must be consistent with equity, not with notional.
    assert abs(r_full.var_dollars - r_full.var_headline * equity) < 1e-6
    assert abs(r_full.gross_leverage - full.sum() / equity) < 1e-9

    print(f"  full book VaR    = {r_full.var_headline:.4%} "
          f"(leverage {r_full.gross_leverage:.2f})")
    print(f"  half book VaR    = {r_half.var_headline:.4%} "
          f"(leverage {r_half.gross_leverage:.2f})")
    print(f"  ratio            = {ratio:.4f}   expected 0.5000  ✓")


# ---------------------------------------------------------------------------

def main() -> int:
    tests = [
        ("Parametric VaR vs analytic Gaussian", test_parametric_var_matches_analytic_gaussian),
        ("Student-t is more conservative", test_student_t_var_is_more_conservative_than_gaussian),
        ("Three estimators agree", test_three_estimators_agree_on_gaussian_data),
        ("CVaR exceeds VaR", test_cvar_exceeds_var),
        ("Beta-weighted delta recovers beta=2", test_beta_weighted_delta_recovers_known_beta),
        ("Benchmark position has unit beta", test_benchmark_position_has_unit_beta),
        ("Kupiec accepts a good model", test_kupiec_accepts_correctly_calibrated_var),
        ("Kupiec rejects an overconfident model", test_kupiec_rejects_an_overconfident_var),
        ("Christoffersen detects clustering", test_christoffersen_detects_clustering),
        ("GARCH recovers simulated parameters", test_garch_recovers_simulated_parameters),
        ("Edge signal fires in both directions", test_edge_signal_directions),
        ("End-to-end risk report", test_full_risk_report),
        ("VaR scales with exposure (regression)", test_var_scales_with_exposure),
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
