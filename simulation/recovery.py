
from simulation.lcr import calculate_lcr


def calculate_required_hqla(
    cash_outflow,
    cash_inflow,
    target_lcr=100
):
    """
    Calculate the HQLA required to achieve a target LCR.

    target_lcr=100 means the project/reference threshold.
    """

    net_cash_outflow = cash_outflow - cash_inflow

    if net_cash_outflow <= 0:
        return 0

    required_hqla = (
        net_cash_outflow * target_lcr / 100
    )

    return required_hqla


def calculate_additional_hqla(
    current_hqla,
    cash_outflow,
    cash_inflow,
    target_lcr=100
):
    """
    Calculate additional HQLA required to reach target LCR.
    """

    required_hqla = calculate_required_hqla(
        cash_outflow=cash_outflow,
        cash_inflow=cash_inflow,
        target_lcr=target_lcr
    )

    additional_hqla = max(
        required_hqla - current_hqla,
        0
    )

    return {
        "current_hqla": current_hqla,
        "required_hqla": required_hqla,
        "additional_hqla": additional_hqla
    }


def simulate_recovery(
    current_hqla,
    cash_outflow,
    cash_inflow,
    additional_hqla=0
):
    """
    Simulate the LCR after adding additional HQLA.
    """

    projected_hqla = (
        current_hqla + additional_hqla
    )

    projected_lcr = calculate_lcr(
        hqla=projected_hqla,
        cash_outflow=cash_outflow,
        cash_inflow=cash_inflow
    )

    return {
        "projected_hqla": projected_hqla,
        "projected_lcr": projected_lcr
    }