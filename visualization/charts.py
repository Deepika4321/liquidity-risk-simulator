
import plotly.express as px
import plotly.graph_objects as go


def lcr_distribution_chart(df):
    """
    Show distribution of LCR across all scenarios.
    """

    fig = px.histogram(
        df,
        x="lcr",
        nbins=50,
        title="LCR Distribution Across Scenarios",
        labels={
            "lcr": "Liquidity Coverage Ratio (%)"
        }
    )

    # Basel/project reference threshold
    fig.add_vline(
        x=100,
        line_dash="dash",
        annotation_text="100% Reference Threshold"
    )

    fig.update_layout(
        template="plotly_white",
        height=450
    )

    return fig


def stress_distribution_chart(df):
    """
    Show number of scenarios in each stress category.
    """

    distribution = (
        df["stress_category"]
        .value_counts()
        .reindex(
            ["Normal", "Moderate", "Severe", "Extreme"],
            fill_value=0
        )
        .reset_index()
    )

    distribution.columns = [
        "stress_category",
        "scenario_count"
    ]

    fig = px.bar(
        distribution,
        x="stress_category",
        y="scenario_count",
        title="Stress Scenario Distribution",
        labels={
            "stress_category": "Stress Category",
            "scenario_count": "Number of Scenarios"
        }
    )

    fig.update_layout(
        template="plotly_white",
        height=400
    )

    return fig


def interest_rate_vs_lcr_chart(df):
    """
    Show relationship between interest rate shock and LCR.
    """

    fig = px.scatter(
        df,
        x="effective_interest_rate",
        y="lcr",
        color="stress_category",
        title="Interest Rate vs LCR",
        labels={
            "effective_interest_rate": "Effective Interest Rate (%)",
            "lcr": "LCR (%)"
        },
        opacity=0.5
    )

    fig.add_hline(
        y=100,
        line_dash="dash",
        annotation_text="100% Reference Threshold"
    )

    fig.update_layout(
        template="plotly_white",
        height=450
    )

    return fig


def withdrawal_vs_lcr_chart(df):
    """
    Show relationship between withdrawal pressure and LCR.
    """

    fig = px.scatter(
        df,
        x="withdrawal_rate",
        y="lcr",
        color="stress_category",
        title="Withdrawal Pressure vs LCR",
        labels={
            "withdrawal_rate": "Withdrawal Rate",
            "lcr": "LCR (%)"
        },
        opacity=0.5
    )

    fig.update_xaxes(tickformat=".0%")

    fig.add_hline(
        y=100,
        line_dash="dash",
        annotation_text="100% Reference Threshold"
    )

    fig.update_layout(
        template="plotly_white",
        height=450
    )

    return fig


def stress_score_vs_lcr_chart(df):
    """
    Show how overall stress score affects LCR.
    """

    fig = px.scatter(
        df,
        x="stress_score",
        y="lcr",
        color="stress_category",
        title="Stress Score vs LCR",
        labels={
            "stress_score": "Stress Score",
            "lcr": "LCR (%)"
        },
        opacity=0.5
    )

    fig.add_hline(
        y=100,
        line_dash="dash",
        annotation_text="100% Reference Threshold"
    )

    fig.update_layout(
        template="plotly_white",
        height=450
    )

    return fig