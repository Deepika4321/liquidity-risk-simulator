from simulation.monte_carlo import run_monte_carlo

from visualization.charts import (
    lcr_distribution_chart,
    stress_distribution_chart,
    interest_rate_vs_lcr_chart,
    withdrawal_vs_lcr_chart,
    stress_score_vs_lcr_chart
)


def test_lcr_distribution_chart():

    df = run_monte_carlo(
        n_simulations=100,
        seed=42
    )

    fig = lcr_distribution_chart(df)

    assert fig is not None


def test_stress_distribution_chart():

    df = run_monte_carlo(
        n_simulations=100,
        seed=42
    )

    fig = stress_distribution_chart(df)

    assert fig is not None


def test_interest_rate_vs_lcr_chart():

    df = run_monte_carlo(
        n_simulations=100,
        seed=42
    )

    fig = interest_rate_vs_lcr_chart(df)

    assert fig is not None


def test_withdrawal_vs_lcr_chart():

    df = run_monte_carlo(
        n_simulations=100,
        seed=42
    )

    fig = withdrawal_vs_lcr_chart(df)

    assert fig is not None


def test_stress_score_vs_lcr_chart():

    df = run_monte_carlo(
        n_simulations=100,
        seed=42
    )

    fig = stress_score_vs_lcr_chart(df)

    assert fig is not None
