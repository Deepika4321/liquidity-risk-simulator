import os

from simulation.monte_carlo import run_monte_carlo

from database.db_manager import (
    save_scenarios,
    get_summary_metrics,
    get_stress_distribution,
    get_worst_scenarios,
    DATABASE_PATH
)


def test_database_workflow():

    # Generate 100 test scenarios
    df = run_monte_carlo(
        n_simulations=100,
        seed=42
    )

    # Save scenarios into DuckDB
    save_scenarios(df)

    # Check summary metrics
    summary = get_summary_metrics()

    assert summary["total_scenarios"] == 100
    assert summary["average_lcr"] > 0
    assert summary["worst_lcr"] > 0

    # Check stress distribution
    distribution = get_stress_distribution()

    assert len(distribution) > 0
    assert distribution["scenario_count"].sum() == 100

    # Check worst scenarios
    worst = get_worst_scenarios(limit=5)

    assert len(worst) == 5

    # Check database file exists
    assert os.path.exists(DATABASE_PATH)
