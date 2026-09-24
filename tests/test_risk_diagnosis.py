from simulation.risk_diagnosis import diagnose_scenario


def test_risk_diagnosis():

    result = diagnose_scenario(
        base_hqla=1200,
        base_outflow=1000,
        base_inflow=250,
        base_interest_rate=6.5,

        scenario_hqla=1080,
        scenario_outflow=1500,
        scenario_inflow=150,

        scenario_interest_rate=10.5,
        scenario_lcr=80
    )

    assert result["scenario_lcr"] == 80

    assert len(result["drivers"]) == 4

    assert result["primary_driver"] in [
        "HQLA Reduction",
        "Outflow Increase",
        "Inflow Reduction",
        "Interest Rate Stress"
    ]

    assert result["primary_driver_impact"] >= 0
