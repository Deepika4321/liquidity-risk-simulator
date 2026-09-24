from simulation.recovery import (
    calculate_required_hqla,
    calculate_additional_hqla,
    simulate_recovery
)


def test_required_hqla():

    required = calculate_required_hqla(
        cash_outflow=1500,
        cash_inflow=150,
        target_lcr=100
    )

    assert required == 1350


def test_additional_hqla():

    result = calculate_additional_hqla(
        current_hqla=1080,
        cash_outflow=1500,
        cash_inflow=150,
        target_lcr=100
    )

    assert result["required_hqla"] == 1350
    assert result["additional_hqla"] == 270


def test_recovery_simulation():

    result = simulate_recovery(
        current_hqla=1080,
        cash_outflow=1500,
        cash_inflow=150,
        additional_hqla=270
    )

    assert result["projected_hqla"] == 1350
    assert result["projected_lcr"] == 100
