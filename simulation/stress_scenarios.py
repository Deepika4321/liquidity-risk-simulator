def apply_stress(
    hqla,
    cash_outflow,
    cash_inflow,
    interest_rate_shock=0.0,
    withdrawal_rate=0.0
):
    """
    Apply synthetic liquidity stress.

    Parameters:
        hqla: HQLA in ₹ Crore
        cash_outflow: expected cash outflow in ₹ Crore
        cash_inflow: expected cash inflow in ₹ Crore
        interest_rate_shock: increase in percentage points
        withdrawal_rate: withdrawal stress as decimal
                           e.g. 0.10 = 10%

    Returns:
        Dictionary containing stressed financial values.
    """

    # -----------------------------------------
    # 1. Increase cash outflow
    # -----------------------------------------
    outflow_multiplier = (
        1
        + withdrawal_rate
        + (interest_rate_shock * 0.05)
    )

    stressed_outflow = cash_outflow * outflow_multiplier

    # -----------------------------------------
    # 2. Reduce cash inflow
    # -----------------------------------------
    inflow_multiplier = (
        1
        - (withdrawal_rate * 0.50)
        - (interest_rate_shock * 0.02)
    )

    # Inflow cannot become negative
    inflow_multiplier = max(inflow_multiplier, 0)

    stressed_inflow = cash_inflow * inflow_multiplier

    # -----------------------------------------
    # 3. Apply HQLA haircut
    # -----------------------------------------
    haircut = (
        withdrawal_rate * 0.20
        + interest_rate_shock * 0.02
    )

    # Maximum synthetic haircut = 50%
    haircut = min(haircut, 0.50)

    stressed_hqla = hqla * (1 - haircut)

    # -----------------------------------------
    # 4. Return stressed values
    # -----------------------------------------
    return {
        "stressed_hqla": stressed_hqla,
        "stressed_outflow": stressed_outflow,
        "stressed_inflow": stressed_inflow,
        "haircut": haircut
    }


def calculate_stress_score(
    withdrawal_rate,
    interest_rate_shock,
    stressed_outflow,
    base_outflow
):
    """
    Calculate project-defined stress score.

    Components:
        50% Withdrawal Stress
        30% Interest Rate Stress
        20% Cash Outflow Stress
    """

    withdrawal_component = min(
        withdrawal_rate / 0.35,
        1.0
    )

    interest_component = min(
        interest_rate_shock / 6.0,
        1.0
    )

    outflow_increase = (
        (stressed_outflow - base_outflow)
        / base_outflow
    )

    outflow_component = min(
        max(outflow_increase / 0.50, 0),
        1.0
    )

    score = (
        0.50 * withdrawal_component
        + 0.30 * interest_component
        + 0.20 * outflow_component
    )

    return score


def classify_stress(stress_score):
    """
    Convert stress score into project-defined category.
    """

    if stress_score < 0.25:
        return "Normal"

    elif stress_score < 0.50:
        return "Moderate"

    elif stress_score < 0.75:
        return "Severe"

    else:
        return "Extreme"