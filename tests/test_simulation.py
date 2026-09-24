from simulation.monte_carlo import run_monte_carlo

from simulation.stress_scenarios import (
    apply_stress,
    calculate_stress_score,
    classify_stress
)


# ============================================
# STRESS SCENARIO TESTS
# ============================================

def test_stress_increases_outflow():

    result = apply_stress(
        hqla=1200,
        cash_outflow=1000,
        cash_inflow=250,
        interest_rate_shock=2,
        withdrawal_rate=0.10
    )

    assert result["stressed_outflow"] > 1000


def test_stress_reduces_inflow():

    result = apply_stress(
        hqla=1200,
        cash_outflow=1000,
        cash_inflow=250,
        interest_rate_shock=2,
        withdrawal_rate=0.10
    )

    assert result["stressed_inflow"] < 250


def test_stress_reduces_hqla():

    result = apply_stress(
        hqla=1200,
        cash_outflow=1000,
        cash_inflow=250,
        interest_rate_shock=2,
        withdrawal_rate=0.10
    )

    assert result["stressed_hqla"] < 1200


def test_stress_score():

    result = calculate_stress_score(
        withdrawal_rate=0.10,
        interest_rate_shock=2,
        stressed_outflow=1200,
        base_outflow=1000
    )

    assert 0 <= result <= 1


def test_stress_classification():

    assert classify_stress(0.10) == "Normal"
    assert classify_stress(0.35) == "Moderate"
    assert classify_stress(0.60) == "Severe"
    assert classify_stress(0.90) == "Extreme"


# ============================================
# MONTE CARLO TESTS
# ============================================

def test_monte_carlo_generates_10000_scenarios():

    df = run_monte_carlo(
        n_simulations=10_000,
        seed=42
    )

    assert len(df) == 10_000


def test_monte_carlo_columns():

    df = run_monte_carlo(
        n_simulations=100,
        seed=42
    )

    required_columns = [
        "scenario_id",
        "interest_rate_shock",
        "withdrawal_rate",
        "hqla",
        "cash_outflow",
        "cash_inflow",
        "net_cash_outflow",
        "lcr",
        "stress_score",
        "stress_category"
    ]

    for column in required_columns:
        assert column in df.columns


def test_lcr_values_are_valid():

    df = run_monte_carlo(
        n_simulations=1000,
        seed=42
    )

    valid_lcr = df["lcr"].dropna()

    assert (valid_lcr >= 0).all()