
from simulation.lcr import (
    calculate_net_cash_outflow,
    calculate_lcr
)


def test_net_cash_outflow():
    result = calculate_net_cash_outflow(1000, 200)

    assert result == 800


def test_lcr():
    result = calculate_lcr(
        hqla=1200,
        cash_outflow=1000,
        cash_inflow=200
    )

    assert result == 150