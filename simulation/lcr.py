def calculate_net_cash_outflow(cash_outflow, cash_inflow):
    """
    Calculate net cash outflow.
    """
    return cash_outflow - cash_inflow


def calculate_lcr(hqla, cash_outflow, cash_inflow):
    """
    Calculate Liquidity Coverage Ratio.

    LCR = HQLA / Net Cash Outflow * 100
    """

    net_cash_outflow = calculate_net_cash_outflow(
        cash_outflow,
        cash_inflow
    )

    if net_cash_outflow <= 0:
        return None

    return (hqla / net_cash_outflow) * 100
