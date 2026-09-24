import duckdb
import pandas as pd


DATABASE_PATH = "data/liquidity_risk.duckdb"


def create_connection():
    return duckdb.connect(DATABASE_PATH)


def save_scenarios(df):
    """
    Save Monte Carlo scenarios into DuckDB.
    """

    con = create_connection()

    # Remove old simulation results
    con.execute("DROP TABLE IF EXISTS scenarios")

    # Register Pandas DataFrame
    con.register("scenario_data", df)

    # Create DuckDB table
    con.execute("""
        CREATE TABLE scenarios AS
        SELECT *
        FROM scenario_data
    """)

    con.close()


def get_summary_metrics():
    """
    Calculate key liquidity risk metrics.
    """

    con = create_connection()

    result = con.execute("""
        SELECT
            COUNT(*) AS total_scenarios,
            AVG(lcr) AS average_lcr,
            MEDIAN(lcr) AS median_lcr,
            MIN(lcr) AS worst_lcr,
            COUNT(
                CASE
                    WHEN lcr < 100 THEN 1
                END
            ) AS scenarios_below_100
        FROM scenarios
        WHERE lcr IS NOT NULL
    """).fetchone()

    con.close()

    total_scenarios = result[0]
    average_lcr = result[1]
    median_lcr = result[2]
    worst_lcr = result[3]
    scenarios_below_100 = result[4]

    percentage_below_100 = (
        scenarios_below_100 / total_scenarios * 100
        if total_scenarios > 0
        else 0
    )

    return {
        "total_scenarios": total_scenarios,
        "average_lcr": average_lcr,
        "median_lcr": median_lcr,
        "worst_lcr": worst_lcr,
        "scenarios_below_100": scenarios_below_100,
        "percentage_below_100": percentage_below_100
    }


def get_stress_distribution():
    """
    Get number of scenarios in each stress category.
    """

    con = create_connection()

    result = con.execute("""
        SELECT
            stress_category,
            COUNT(*) AS scenario_count
        FROM scenarios
        GROUP BY stress_category
        ORDER BY
            CASE stress_category
                WHEN 'Normal' THEN 1
                WHEN 'Moderate' THEN 2
                WHEN 'Severe' THEN 3
                WHEN 'Extreme' THEN 4
            END
    """).fetchdf()

    con.close()

    return result


def get_worst_scenarios(limit=10):
    """
    Return scenarios with the lowest LCR.
    """

    con = create_connection()

    result = con.execute("""
        SELECT
            scenario_id,
            effective_interest_rate,
            withdrawal_rate,
            hqla,
            cash_outflow,
            cash_inflow,
            net_cash_outflow,
            lcr,
            stress_score,
            stress_category
        FROM scenarios
        WHERE lcr IS NOT NULL
        ORDER BY lcr ASC
        LIMIT ?
    """, [limit]).fetchdf()

    con.close()

    return result