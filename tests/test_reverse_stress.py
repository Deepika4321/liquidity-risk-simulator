
from simulation.reverse_stress import (
    calculate_lcr_under_stress,
    find_withdrawal_threshold,
    find_interest_rate_threshold
)


def test_lcr_under_stress():

    lcr = calculate_lcr_under_stress(
        hqla=1200,
        cash_outflow=1000,
        cash_inflow=250,
        interest_rate_shock=2,
        withdrawal_rate=0.10
    )

    assert lcr is not None
    assert lcr > 0


def test_withdrawal_threshold():

    result = find_withdrawal_threshold()

    assert result["threshold_found"] is True
    assert result["withdrawal_rate"] > 0
    assert result["lcr"] < 100


def test_interest_rate_threshold():

    result = find_interest_rate_threshold()

    assert result["threshold_found"] is True
    assert result["interest_rate_shock"] > 0
    assert result["lcr"] < 100