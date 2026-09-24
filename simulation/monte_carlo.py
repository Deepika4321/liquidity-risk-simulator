import numpy as np
import pandas as pd

from simulation.lcr import calculate_net_cash_outflow
from simulation.stress_scenarios import (
    apply_stress,
    calculate_stress_score,
    classify_stress
)


def run_monte_carlo(
    hqla=1200,
    cash_outflow=1000,
    cash_inflow=250,
    base_interest_rate=6.5,
    base_withdrawal_rate=0.05,
    n_simulations=10_000,
    seed=42
):
    """
    Generate Monte Carlo liquidity stress scenarios.

    Returns:
        pandas DataFrame containing all simulated scenarios.
    """

    rng = np.random.default_rng(seed)

    # -----------------------------------------
    # 1. Generate common stress factor
    # -----------------------------------------
    stress_factor = rng.beta(
        2,
        5,
        n_simulations
    )

    # -----------------------------------------
    # 2. Generate interest-rate shocks
    # Maximum project shock = 6 percentage points
    # -----------------------------------------
    interest_rate_shock = (
        stress_factor * 6
        + rng.normal(0, 0.25, n_simulations)
    )

    interest_rate_shock = np.clip(
        interest_rate_shock,
        0,
        6
    )

    # -----------------------------------------
    # 3. Generate withdrawal stress
    # Maximum = 35%
    # -----------------------------------------
    withdrawal_rate = (
        base_withdrawal_rate
        + stress_factor * 0.30
        + rng.normal(0, 0.01, n_simulations)
    )

    withdrawal_rate = np.clip(
        withdrawal_rate,
        0,
        0.35
    )

    # -----------------------------------------
    # 4. Vectorized financial calculations
    # -----------------------------------------

    outflow_multiplier = (
        1
        + withdrawal_rate
        + (interest_rate_shock * 0.05)
    )

    stressed_outflow = (
        cash_outflow
        * outflow_multiplier
    )

    inflow_multiplier = (
        1
        - (withdrawal_rate * 0.50)
        - (interest_rate_shock * 0.02)
    )

    inflow_multiplier = np.clip(
        inflow_multiplier,
        0,
        None
    )

    stressed_inflow = (
        cash_inflow
        * inflow_multiplier
    )

    # -----------------------------------------
    # 5. HQLA haircut
    # -----------------------------------------

    haircut = (
        withdrawal_rate * 0.20
        + interest_rate_shock * 0.02
    )

    haircut = np.clip(
        haircut,
        0,
        0.50
    )

    stressed_hqla = (
        hqla * (1 - haircut)
    )

    # -----------------------------------------
    # 6. Net cash outflow
    # -----------------------------------------

    net_cash_outflow = (
        stressed_outflow
        - stressed_inflow
    )

    # -----------------------------------------
    # 7. LCR
    # -----------------------------------------

    lcr = np.where(
        net_cash_outflow > 0,
        (stressed_hqla / net_cash_outflow) * 100,
        np.nan
    )

    # -----------------------------------------
    # 8. Stress score
    # -----------------------------------------

    withdrawal_component = np.clip(
        withdrawal_rate / 0.35,
        0,
        1
    )

    interest_component = np.clip(
        interest_rate_shock / 6,
        0,
        1
    )

    outflow_increase = (
        (stressed_outflow - cash_outflow)
        / cash_outflow
    )

    outflow_component = np.clip(
        outflow_increase / 0.50,
        0,
        1
    )

    stress_score = (
        0.50 * withdrawal_component
        + 0.30 * interest_component
        + 0.20 * outflow_component
    )

    # -----------------------------------------
    # 9. Stress category
    # -----------------------------------------

    stress_category = np.select(
        [
            stress_score < 0.25,
            stress_score < 0.50,
            stress_score < 0.75
        ],
        [
            "Normal",
            "Moderate",
            "Severe"
        ],
        default="Extreme"
    )

    # -----------------------------------------
    # 10. Create scenario DataFrame
    # -----------------------------------------

    scenarios = pd.DataFrame({
        "scenario_id": np.arange(
            1,
            n_simulations + 1
        ),

        "base_interest_rate": base_interest_rate,

        "interest_rate_shock": interest_rate_shock,

        "effective_interest_rate": (
            base_interest_rate
            + interest_rate_shock
        ),

        "withdrawal_rate": withdrawal_rate,

        "hqla": stressed_hqla,

        "cash_outflow": stressed_outflow,

        "cash_inflow": stressed_inflow,

        "net_cash_outflow": net_cash_outflow,

        "lcr": lcr,

        "stress_score": stress_score,

        "stress_category": stress_category
    })

    return scenarios
