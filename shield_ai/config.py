"""
Shield-AI — central configuration.

Every number the agent is allowed to act on lives here, not scattered through
the code. This matters for the hackathon: a judge can read one file and see
exactly what the risk mandate is, and you can change the agent's behaviour
without touching logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Alpaca credentials — loaded from the environment, never hard-coded.
# Copy .env.example to .env and fill it in. .env is git-ignored.
# --------------------------------------------------------------------------

ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_PAPER: bool = os.getenv("ALPACA_PAPER_TRADE", "true").lower() == "true"


@dataclass(frozen=True)
class RiskMandate:
    """
    The hard boundary the language model is not allowed to cross.

    Everything here is deterministic. The LLM proposes; these numbers dispose.
    Frozen because a mandate that can be mutated at runtime is not a mandate.
    """

    # --- Value-at-Risk budget -------------------------------------------------
    confidence: float = 0.99          # 99% one-day VaR
    horizon_days: int = 1
    var_limit_pct: float = 0.025      # breach if 1-day 99% VaR > 2.5% of equity
    var_warn_pct: float = 0.020       # amber zone: start looking for a hedge

    # --- Directional exposure -------------------------------------------------
    # Beta-weighted net delta, expressed in units of underlying-equivalent
    # exposure divided by account equity. 1.0 == fully long the market.
    delta_band: tuple[float, float] = (0.30, 0.85)

    # --- Estimation windows ---------------------------------------------------
    lookback_days: int = 252          # one trading year for the covariance matrix
    min_observations: int = 60        # refuse to estimate on thinner history
    mc_paths: int = 50_000            # Monte Carlo simulation paths

    # --- Fat tails ------------------------------------------------------------
    # Equity returns are not Gaussian. A Student-t with ~5 df is the standard
    # compromise: heavy enough to respect crash risk, light enough to estimate
    # stably on one year of data.
    student_t_df: float = 5.0

    # --- Premium-selling edge (IV vs GARCH forecast) --------------------------
    min_edge_vol_points: float = 4.0  # sell only if IV exceeds forecast by 4 vol pts
    target_short_delta: float = 0.20  # ~20-delta strikes
    min_dte: int = 25
    max_dte: int = 45
    take_profit_pct: float = 0.50     # buy back after capturing 50% of premium
    roll_at_dte: int = 21

    # --- Hard safety rails ----------------------------------------------------
    max_hedge_cost_pct: float = 0.015     # never spend >1.5% of equity on a hedge
    max_positions: int = 8
    allow_naked_short_calls: bool = False  # never. covered or cash-secured only.

    universe: tuple[str, ...] = field(
        default=("SPY", "AAPL", "MSFT", "NVDA", "JPM", "XOM")
    )
    benchmark: str = "SPY"


MANDATE = RiskMandate()
