
def diagnose_scenario(
    base_hqla,
    base_outflow,
    base_inflow,
    base_interest_rate,
    scenario_hqla,
    scenario_outflow,
    scenario_inflow,
    scenario_interest_rate,
    scenario_lcr
):
    """
    Diagnose the major drivers of liquidity deterioration.

    Uses percentage changes between base and stressed
    financial conditions.
    """

    # HQLA deterioration
    hqla_change = (
        (base_hqla - scenario_hqla)
        / base_hqla
        if base_hqla > 0
        else 0
    )

    # Outflow increase
    outflow_change = (
        (scenario_outflow - base_outflow)
        / base_outflow
        if base_outflow > 0
        else 0
    )

    # Inflow deterioration
    inflow_change = (
        (base_inflow - scenario_inflow)
        / base_inflow
        if base_inflow > 0
        else 0
    )

    # Interest rate increase
    rate_change = max(
        scenario_interest_rate - base_interest_rate,
        0
    )

    drivers = {
        "HQLA Reduction": max(hqla_change, 0),
        "Outflow Increase": max(outflow_change, 0),
        "Inflow Reduction": max(inflow_change, 0),
        "Interest Rate Stress": rate_change / 6
    }

    # Sort from largest driver to smallest
    ranked_drivers = sorted(
        drivers.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "scenario_lcr": scenario_lcr,
        "drivers": ranked_drivers,
        "primary_driver": ranked_drivers[0][0],
        "primary_driver_impact": ranked_drivers[0][1]
    }