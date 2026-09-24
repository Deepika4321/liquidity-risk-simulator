
from simulation.stress_scenarios import apply_stress
from simulation.lcr import calculate_lcr


def calculate_lcr_under_stress(
    hqla,
    cash_outflow,
    cash_inflow,
    interest_rate_shock,
    withdrawal_rate
):
    """
    Calculate LCR for a given stress scenario.
    """

    stressed = apply_stress(
        hqla=hqla,
        cash_outflow=cash_outflow,
        cash_inflow=cash_inflow,
        interest_rate_shock=interest_rate_shock,
        withdrawal_rate=withdrawal_rate
    )

    lcr = calculate_lcr(
        hqla=stressed["stressed_hqla"],
        cash_outflow=stressed["stressed_outflow"],
        cash_inflow=stressed["stressed_inflow"]
    )

    return lcr


def find_withdrawal_threshold(
    hqla=1200,
    cash_outflow=1000,
    cash_inflow=250,
    interest_rate_shock=0,
    target_lcr=100,
    max_withdrawal=0.35,
    steps=350
):
    """
    Find approximate withdrawal rate at which
    LCR falls below the target threshold.
    """

    for i in range(steps + 1):

        withdrawal_rate = (
            max_withdrawal * i / steps
        )

        lcr = calculate_lcr_under_stress(
            hqla,
            cash_outflow,
            cash_inflow,
            interest_rate_shock,
            withdrawal_rate
        )

        if lcr is not None and lcr < target_lcr:
            return {
                "threshold_found": True,
                "withdrawal_rate": withdrawal_rate,
                "lcr": lcr
            }

    return {
        "threshold_found": False,
        "withdrawal_rate": None,
        "lcr": None
    }


def find_interest_rate_threshold(
    hqla=1200,
    cash_outflow=1000,
    cash_inflow=250,
    withdrawal_rate=0,
    target_lcr=100,
    max_interest_rate_shock=6,
    steps=300
):
    """
    Find approximate interest-rate shock at which
    LCR falls below the target threshold.
    """

    for i in range(steps + 1):

        interest_rate_shock = (
            max_interest_rate_shock * i / steps
        )

        lcr = calculate_lcr_under_stress(
            hqla,
            cash_outflow,
            cash_inflow,
            interest_rate_shock,
            withdrawal_rate
        )

        if lcr is not None and lcr < target_lcr:
            return {
                "threshold_found": True,
                "interest_rate_shock": interest_rate_shock,
                "lcr": lcr
            }

    return {
        "threshold_found": False,
        "interest_rate_shock": None,
        "lcr": None
    }