import streamlit as st
import pandas as pd

from simulation.monte_carlo import run_monte_carlo
from simulation.reverse_stress import (
    find_withdrawal_threshold,
    find_interest_rate_threshold
)
from simulation.recovery import (
    calculate_additional_hqla,
    simulate_recovery
)
from simulation.risk_diagnosis import diagnose_scenario

from database.db_manager import (
    save_scenarios,
    get_summary_metrics
)

from visualization.charts import (
    lcr_distribution_chart,
    stress_distribution_chart,
    interest_rate_vs_lcr_chart,
    withdrawal_vs_lcr_chart,
    stress_score_vs_lcr_chart
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Liquidity Risk Simulator",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
:root {
    --navy:#0B1F3A; --navy2:#12365F; --blue:#2563EB; --teal:#0F766E;
    --ink:#172033; --muted:#64748B; --line:#E2E8F0; --surface:#FFFFFF;
}
.stApp { background:linear-gradient(180deg,#F8FAFD 0%,#F3F7FB 100%); color:var(--ink); }
[data-testid="stAppViewContainer"] { background:#F5F8FC; }
[data-testid="stHeader"] { background:rgba(245,248,252,.9); }
.block-container { padding-top:1.2rem; padding-bottom:2rem; max-width:1500px; }

[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0B1F3A 0%,#102A4C 100%);
    border-right:1px solid rgba(255,255,255,.08);
}
[data-testid="stSidebar"] * { color:#F8FAFC !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stMarkdown p { color:#CBD5E1 !important; }
[data-testid="stSidebar"] input {
    background:#FFFFFF !important;
    border:1px solid #CBD5E1 !important;
    color:#172033 !important;
    -webkit-text-fill-color:#172033 !important;
    border-radius:10px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background:#FFFFFF !important;
    border:1px solid #CBD5E1 !important;
    color:#172033 !important;
    border-radius:10px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span {
    color:#172033 !important;
    -webkit-text-fill-color:#172033 !important;
}

/* Keep the Number of Scenarios selectbox text dark and readable. */
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] div[role="button"] {
    color:#172033 !important;
    -webkit-text-fill-color:#172033 !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button {
    background:linear-gradient(135deg,#14B8A6,#2563EB) !important;
    border:0 !important; color:white !important; font-weight:800 !important;
    border-radius:10px !important; box-shadow:0 8px 20px rgba(37,99,235,.22);
}

.hero-card {
    background:linear-gradient(135deg,#0B1F3A 0%,#12365F 58%,#0F766E 130%);
    border-radius:20px; padding:28px 32px; margin:4px 0 18px;
    color:white; box-shadow:0 10px 28px rgba(11,31,58,.12);
    position:relative; overflow:hidden;
}
.hero-card:after {
    content:""; position:absolute; width:270px; height:270px; right:-100px; top:-130px;
    border-radius:50%; background:rgba(20,184,166,.20);
}
.hero-kicker { color:#7DD3FC; font-size:.76rem; font-weight:800; letter-spacing:.12em; margin-bottom:7px; }
.hero-title { font-size:1.9rem; line-height:1.15; font-weight:800; margin:0 0 8px; color:#fff; }
.hero-subtitle { color:#D9E7F5; font-size:.98rem; line-height:1.55; max-width:920px; margin:0; }
.hero-pills { display:flex; flex-wrap:wrap; gap:8px; margin-top:17px; }
.hero-pill {
    background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.14);
    border-radius:999px; padding:6px 11px; color:#E6F2FF; font-size:.76rem; font-weight:650;
}
.section-title { color:var(--navy); font-size:1.35rem; font-weight:800; margin:8px 0 4px; }
.section-subtitle { color:var(--muted); margin-bottom:16px; }

.metric-card {
    background:#fff; border:1px solid var(--line); border-radius:15px;
    padding:16px 18px; box-shadow:0 5px 18px rgba(15,23,42,.05); min-height:105px;
}
.metric-label { color:var(--muted); font-size:.76rem; font-weight:750; text-transform:uppercase; letter-spacing:.04em; }
.metric-value { color:var(--navy); font-size:1.55rem; font-weight:850; margin-top:6px; }
.metric-note { color:#94A3B8; font-size:.72rem; margin-top:3px; }

.risk-safe,.risk-warning,.risk-danger { padding:15px 18px; border-radius:12px; margin:14px 0 20px; font-size:.92rem; }
.risk-safe { background:#ECFDF5; border:1px solid #A7F3D0; border-left:5px solid #10B981; color:#065F46; }
.risk-warning { background:#FFFBEB; border:1px solid #FDE68A; border-left:5px solid #F59E0B; color:#92400E; }
.risk-danger { background:#FEF2F2; border:1px solid #FECACA; border-left:5px solid #EF4444; color:#991B1B; }

button[data-baseweb="tab"] { font-weight:750 !important; color:#64748B !important; padding:10px 11px !important; }
button[data-baseweb="tab"][aria-selected="true"] { color:#0B1F3A !important; }
div[data-baseweb="tab-highlight"] { background:linear-gradient(90deg,#14B8A6,#2563EB) !important; height:3px !important; }

.stButton > button,.stDownloadButton > button {
    border-radius:10px !important; font-weight:700 !important;
    border:1px solid #D7E0EA !important; background:#fff !important; color:#17304F !important;
}
.stButton > button:hover,.stDownloadButton > button:hover {
    border-color:#2563EB !important; color:#2563EB !important;
    box-shadow:0 5px 15px rgba(37,99,235,.10);
}
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
[data-baseweb="select"] > div,[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input {
    border-radius:9px !important; border-color:#D7E0EA !important;
}
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:12px; background:#fff; }

.info-strip {
    display:flex; align-items:center; gap:10px; background:#EFF6FF; border:1px solid #BFDBFE;
    color:#1E40AF; padding:11px 14px; border-radius:10px; font-size:.84rem; margin:8px 0 18px;
}
.footer { color:#94A3B8; text-align:center; font-size:.78rem; padding:12px 0; }

/* Visible +/- controls for baseline number inputs */
[data-testid="stSidebar"] [data-testid="stNumberInput"] {
    background: transparent !important;
}

[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    background: #0B1F3A !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 0 !important;
    border-left: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 0 !important;
    min-width: 34px !important;
    min-height: 38px !important;
    height: 38px !important;
    font-size: 18px !important;
    font-weight: 800 !important;
    opacity: 1 !important;
    box-shadow: none !important;
}

[data-testid="stSidebar"] [data-testid="stNumberInput"] button:hover {
    background: #163A63 !important;
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-testid="stNumberInput"] button:active {
    background: #1E4B7A !important;
}

/* Streamlit renders +/- as SVG icons. Force the icon to white. */
[data-testid="stSidebar"] [data-testid="stNumberInput"] button svg {
    width: 16px !important;
    height: 16px !important;
    color: #FFFFFF !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] [data-testid="stNumberInput"] button svg path,
[data-testid="stSidebar"] [data-testid="stNumberInput"] button svg line,
[data-testid="stSidebar"] [data-testid="stNumberInput"] button svg polyline {
    fill: #FFFFFF !important;
    stroke: #FFFFFF !important;
    color: #FFFFFF !important;
    opacity: 1 !important;
}

/* Keep the input itself clean white. */
[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    background: #FFFFFF !important;
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
    font-weight: 600 !important;
    border: 0 !important;
}

/* Disabled state is still visible, but slightly muted. */
[data-testid="stSidebar"] [data-testid="stNumberInput"] button:disabled {
    background: #17304F !important;
    color: #CBD5E1 !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] [data-testid="stNumberInput"] button:disabled svg,
[data-testid="stSidebar"] [data-testid="stNumberInput"] button:disabled svg path {
    color: #CBD5E1 !important;
    fill: #CBD5E1 !important;
    stroke: #CBD5E1 !important;
    opacity: 1 !important;
}

/* Make the withdrawal slider feel more obviously interactive */
[data-testid="stSidebar"] [data-testid="stSlider"] {
    padding-top: 2px;
    padding-bottom: 4px;
}


/* Unified baseline input appearance */
[data-testid="stSidebar"] [data-testid="stNumberInput"] > div {
    border-radius: 9px !important;
    overflow: hidden !important;
    border: 1px solid #CBD5E1 !important;
    background: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-testid="stNumberInput"] [data-baseweb="input"] {
    border: 0 !important;
    box-shadow: none !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Simulation Controls")

st.sidebar.markdown(
    "Configure the synthetic bank liquidity environment."
)
st.sidebar.markdown(
    '<div style="font-size:.78rem;font-weight:800;letter-spacing:.08em;'
    'text-transform:uppercase;color:#7DD3FC !important;margin:18px 0 8px;">'
    'Baseline Parameters</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div style="background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12);
    border-radius:12px;padding:12px 13px;margin:14px 0 18px;">
      <div style="font-weight:800;color:#FFFFFF;margin-bottom:6px;">⚡ Quick Guide</div>
      <div style="font-size:.78rem;line-height:1.55;color:#CBD5E1;">
        <b style="color:#FFFFFF;">1.</b> Set baseline assumptions<br>
        <b style="color:#FFFFFF;">2.</b> Run 1K / 5K / 10K scenarios<br>
        <b style="color:#FFFFFF;">3.</b> Explore risk & thresholds<br>
        <b style="color:#FFFFFF;">4.</b> Test recovery actions
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

hqla = st.sidebar.number_input(
    "HQLA (₹ Crore)",
    min_value=100.0,
    value=1200.0,
    step=50.0
)

cash_outflow = st.sidebar.number_input(
    "Base Cash Outflow (₹ Crore)",
    min_value=100.0,
    value=1000.0,
    step=50.0
)

cash_inflow = st.sidebar.number_input(
    "Base Cash Inflow (₹ Crore)",
    min_value=0.0,
    value=250.0,
    step=25.0
)

base_interest_rate = st.sidebar.number_input(
    "Base Interest Rate (%)",
    min_value=0.0,
    value=6.5,
    step=0.25
)
base_withdrawal_percent = st.sidebar.slider(
    "Base Withdrawal Pressure",
    min_value=0,
    max_value=35,
    value=5,
    step=1,
    format="%d%%"
)

base_withdrawal_rate = base_withdrawal_percent / 100

st.sidebar.markdown(
    '<div style="font-size:.73rem;color:#94A3B8 !important;margin:-2px 0 10px;">'
    'These values define the starting point for the Monte Carlo stress simulation.'
    '</div>',
    unsafe_allow_html=True
)

n_simulations = st.sidebar.selectbox(
    "Number of Scenarios",
    [1000, 5000, 10000],
    index=2
)

run_button = st.sidebar.button(
    "🚀 Run Stress Simulation",
    use_container_width=True
)

st.sidebar.markdown(
    """
    <div style="
        margin-top:10px;
        padding:9px 11px;
        border-left:3px solid #14B8A6;
        background:rgba(20,184,166,.08);
        color:#CBD5E1;
        border-radius:0 8px 8px 0;
        font-size:.74rem;
        line-height:1.45;
    ">
        <b style="color:#FFFFFF;">Interactive:</b>
        use the <b>− / +</b> controls or sliders to change assumptions,
        then run the simulation to update every chart and risk metric.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIMULATION
# ============================================================

if "simulation_df" not in st.session_state:
    run_button = True


if run_button:

    with st.spinner("Running Monte Carlo stress simulation..."):

        df = run_monte_carlo(
            hqla=hqla,
            cash_outflow=cash_outflow,
            cash_inflow=cash_inflow,
            base_interest_rate=base_interest_rate,
            base_withdrawal_rate=base_withdrawal_rate,
            n_simulations=n_simulations,
            seed=42
        )

        save_scenarios(df)

        st.session_state["simulation_df"] = df


df = st.session_state["simulation_df"]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-kicker">FINANCIAL RISK INTELLIGENCE</div>
        <div class="hero-title">Liquidity Risk Decision Platform</div>
        <p class="hero-subtitle">
            Real-time liquidity stress testing powered by Monte Carlo simulation.
            Generate 10,000 synthetic financial futures, measure LCR deterioration,
            identify critical stress thresholds, diagnose risk drivers, and test recovery actions.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="info-strip">ℹ️ Synthetic hackathon model • ₹ Crore • 100% is the project/reference threshold • Not real bank data</div>',
    unsafe_allow_html=True
)

# SUMMARY METRICS
# ============================================================

summary = get_summary_metrics()

total_scenarios = summary["total_scenarios"]
average_lcr = summary["average_lcr"]
median_lcr = summary["median_lcr"]
worst_lcr = summary["worst_lcr"]
percentage_below = summary["percentage_below_100"]


st.markdown('<div class="section-title">Executive Risk Snapshot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">A quick view of the simulated liquidity position across all generated futures.</div>',
    unsafe_allow_html=True
)

metric_cols = st.columns(5)
metric_items = [
    ("Scenarios", f"{total_scenarios:,}", "Generated futures"),
    ("Average LCR", f"{average_lcr:.1f}%", "Portfolio-level average"),
    ("Median LCR", f"{median_lcr:.1f}%", "Middle simulated outcome"),
    ("Worst LCR", f"{worst_lcr:.1f}%", "Lowest simulated outcome"),
    ("Below 100%", f"{percentage_below:.1f}%", "Scenarios under threshold"),
]

for col, (label, value, note) in zip(metric_cols, metric_items):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# RISK STATUS
# ============================================================
# RISK STATUS
# ============================================================

if average_lcr >= 100 and percentage_below < 10:

    st.markdown(
        """
        <div class="risk-safe">
        <b>Liquidity Status:</b> Most simulated scenarios remain above
        the project/reference threshold.
        </div>
        """,
        unsafe_allow_html=True
    )

elif average_lcr >= 100:

    st.markdown(
        """
        <div class="risk-warning">
        <b>Liquidity Status:</b> Average simulated LCR remains above the
        reference threshold, but a meaningful subset of scenarios falls below it.
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
        <div class="risk-danger">
        <b>Liquidity Status:</b> The simulated environment produces an
        average LCR below the project/reference threshold.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TABS
# ============================================================

overview_tab, stress_tab, explorer_tab, diagnosis_tab, reverse_tab, recovery_tab = st.tabs(
    [
        "📈 Overview",
        "🔥 Stress Lab",
        "🔎 Scenarios",
        "🧠 Risk Diagnosis",
        "🎯 Reverse Stress",
        "🛡️ Recovery"
    ]
)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

with overview_tab:

    st.markdown(
        '<div class="section-title">📊 Liquidity Distribution</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-subtitle">See how LCR and stress severity are distributed across the simulated futures.</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.plotly_chart(
            lcr_distribution_chart(df),
            use_container_width=True
        )

    with col2:

        st.plotly_chart(
            stress_distribution_chart(df),
            use_container_width=True
        )

    st.markdown("### Stress Relationships")

    col1, col2 = st.columns(2)

    with col1:

        st.plotly_chart(
            interest_rate_vs_lcr_chart(df),
            use_container_width=True
        )

    with col2:

        st.plotly_chart(
            withdrawal_vs_lcr_chart(df),
            use_container_width=True
        )


# ============================================================
# STRESS LAB
# ============================================================

with stress_tab:

    st.markdown('<div class="section-title">🔥 Stress Lab</div>', unsafe_allow_html=True)

    st.write(
        "Explore how simulated financial stress changes liquidity conditions."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.plotly_chart(
            stress_score_vs_lcr_chart(df),
            use_container_width=True
        )

    with col2:

        st.markdown("#### Scenario Statistics")

        stress_counts = (
            df["stress_category"]
            .value_counts()
            .reindex(
                ["Normal", "Moderate", "Severe", "Extreme"],
                fill_value=0
            )
        )

        for category, count in stress_counts.items():

            percentage = count / len(df) * 100

            st.write(
                f"**{category}:** "
                f"{count:,} scenarios "
                f"({percentage:.1f}%)"
            )

    st.markdown("### Model Inputs")

    input_col1, input_col2, input_col3 = st.columns(3)

    input_col1.metric(
        "Base HQLA",
        f"₹{hqla:,.0f} Cr"
    )

    input_col2.metric(
        "Base Outflow",
        f"₹{cash_outflow:,.0f} Cr"
    )

    input_col3.metric(
        "Base Inflow",
        f"₹{cash_inflow:,.0f} Cr"
    )


# ============================================================
# SCENARIO EXPLORER
# ============================================================

with explorer_tab:

    st.markdown('<div class="section-title">🔎 Scenario Explorer</div>', unsafe_allow_html=True)

    category_filter = st.multiselect(
        "Stress Category",
        ["Normal", "Moderate", "Severe", "Extreme"],
        default=["Normal", "Moderate", "Severe", "Extreme"]
    )

    filtered_df = df[
        df["stress_category"].isin(category_filter)
    ]

    lcr_limit = st.slider(
        "Maximum LCR to display",
        min_value=0,
        max_value=300,
        value=300,
        help="Filter scenarios whose LCR is at or below this value. 300% keeps the full simulated range visible."
    )

    explorer_df = filtered_df[
        filtered_df["lcr"] <= lcr_limit
    ].copy()

    display_columns = [
        "scenario_id",
        "effective_interest_rate",
        "withdrawal_rate",
        "hqla",
        "cash_outflow",
        "cash_inflow",
        "net_cash_outflow",
        "lcr",
        "stress_score",
        "stress_category"
    ]

    # ------------------------------------------------------------
    # Scenario search + complete dataset controls
    # ------------------------------------------------------------

    search_col1, search_col2 = st.columns([2, 1])

    with search_col1:
        scenario_search = st.text_input(
            "🔎 Search Scenario ID",
            placeholder="Example: 5293",
            help="Enter a scenario ID to inspect one specific simulated future."
        )

    with search_col2:
        rows_per_page = st.selectbox(
            "Rows per page",
            [25, 50, 100, 250, 500],
            index=1
        )

    if scenario_search.strip():
        if scenario_search.strip().isdigit():
            scenario_id = int(scenario_search.strip())
            explorer_df = explorer_df[
                explorer_df["scenario_id"] == scenario_id
            ]
        else:
            st.warning("Please enter a numeric Scenario ID.")

    # Sort by LCR so the most stressed matching scenarios appear first.
    explorer_df = explorer_df.sort_values(
        "lcr",
        ascending=True
    ).reset_index(drop=True)

    total_matching = len(explorer_df)

    st.markdown(
        f"**Showing {total_matching:,} matching scenarios** "
        f"out of **{len(df):,} generated scenarios**."
    )

    # Complete 10,000-row CSV download. This is independent of the
    # current filters, so judges can download the entire simulation.
    all_scenarios_csv = df[display_columns].to_csv(index=False).encode("utf-8")

    st.download_button(
        label=f"📥 Download All {len(df):,} Scenarios (CSV)",
        data=all_scenarios_csv,
        file_name=f"liquidity_scenarios_{len(df)}.csv",
        mime="text/csv",
        use_container_width=True
    )

    # ------------------------------------------------------------
    # Pagination
    # ------------------------------------------------------------

    if total_matching > 0:

        total_pages = max(
            1,
            (total_matching + rows_per_page - 1) // rows_per_page
        )

        # ------------------------------------------------------------
        # Reliable pagination state
        # ------------------------------------------------------------
        # Streamlit reruns the script whenever a button is clicked.
        # Keep the current page in session_state and synchronize it with
        # the page-number input BEFORE that widget is created.
        if "explorer_page" not in st.session_state:
            st.session_state["explorer_page"] = 1

        # If filters/search reduced the result set, keep the page valid.
        st.session_state["explorer_page"] = min(
            max(1, int(st.session_state["explorer_page"])),
            total_pages
        )

        page_col1, page_col2, page_col3 = st.columns([1, 2, 1])

        with page_col1:
            previous_page = st.button(
                "← Previous",
                disabled=(st.session_state["explorer_page"] <= 1),
                use_container_width=True,
                key="explorer_previous"
            )

        with page_col3:
            next_page = st.button(
                "Next →",
                disabled=(st.session_state["explorer_page"] >= total_pages),
                use_container_width=True,
                key="explorer_next"
            )

        # Button clicks happen before the page-number widget is created,
        # so updating both state values here makes navigation reliable.
        if previous_page:
            st.session_state["explorer_page"] = max(
                1,
                st.session_state["explorer_page"] - 1
            )

        if next_page:
            st.session_state["explorer_page"] = min(
                total_pages,
                st.session_state["explorer_page"] + 1
            )

        # Synchronize the number input with the button-selected page.
        st.session_state["explorer_page_input"] = st.session_state["explorer_page"]

        with page_col2:
            page_number = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                step=1,
                key="explorer_page_input",
                label_visibility="collapsed"
            )

        # If the user manually enters a page number, remember it.
        st.session_state["explorer_page"] = int(page_number)

        current_page = st.session_state["explorer_page"]

        start_idx = (current_page - 1) * rows_per_page
        end_idx = start_idx + rows_per_page

        page_df = explorer_df.iloc[start_idx:end_idx]

        st.caption(
            f"Page {current_page} of {total_pages} • "
            f"Rows {start_idx + 1:,}–{min(end_idx, total_matching):,}"
        )

        st.dataframe(
            page_df[display_columns],
            use_container_width=True,
            hide_index=True,
            height=520
        )

    else:
        st.info("No scenarios match the selected filters.")

    st.markdown("### ⚠️ Worst 10 Scenarios")

    worst_10 = (
        df[
            display_columns
        ]
        .sort_values("lcr")
        .head(10)
    )

    st.dataframe(
        worst_10,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# RISK DIAGNOSIS
# ============================================================

with diagnosis_tab:

    st.markdown('<div class="section-title">🧠 Why Did LCR Fall?</div>', unsafe_allow_html=True)

    worst_scenario = (
        df[
            df["lcr"].notna()
        ]
        .sort_values("lcr")
        .iloc[0]
    )

    diagnosis = diagnose_scenario(
        base_hqla=hqla,
        base_outflow=cash_outflow,
        base_inflow=cash_inflow,
        base_interest_rate=base_interest_rate,

        scenario_hqla=worst_scenario["hqla"],
        scenario_outflow=worst_scenario["cash_outflow"],
        scenario_inflow=worst_scenario["cash_inflow"],
        scenario_interest_rate=worst_scenario["effective_interest_rate"],

        scenario_lcr=worst_scenario["lcr"]
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Worst Simulated LCR",
            f"{diagnosis['scenario_lcr']:.1f}%"
        )

        st.metric(
            "Primary Driver",
            diagnosis["primary_driver"]
        )

    with col2:

        st.write("### Driver Analysis")

        driver_df = pd.DataFrame(
            diagnosis["drivers"],
            columns=["Driver", "Impact"]
        )

        driver_df["Impact"] = (
            driver_df["Impact"] * 100
        ).round(2)

        driver_df = driver_df.rename(
            columns={
                "Impact": "Relative Change (%)"
            }
        )

        st.dataframe(
            driver_df,
            use_container_width=True,
            hide_index=True
        )

    st.info(
        "Driver ranking is based on the model's simulated changes from "
        "the configured baseline; it is not an empirical attribution."
    )


# ============================================================
# REVERSE STRESS
# ============================================================

with reverse_tab:

    st.markdown('<div class="section-title">🎯 Reverse Stress Testing</div>', unsafe_allow_html=True)

    st.write(
        "Find the approximate stress point where simulated LCR falls below 100%."
    )

    col1, col2 = st.columns(2)

    withdrawal_result = find_withdrawal_threshold(
        hqla=hqla,
        cash_outflow=cash_outflow,
        cash_inflow=cash_inflow,
        interest_rate_shock=0,
        target_lcr=100
    )

    interest_result = find_interest_rate_threshold(
        hqla=hqla,
        cash_outflow=cash_outflow,
        cash_inflow=cash_inflow,
        withdrawal_rate=base_withdrawal_rate,
        target_lcr=100
    )

    with col1:

        st.markdown("#### Withdrawal Threshold")

        if withdrawal_result["threshold_found"]:

            st.metric(
                "Critical Withdrawal Pressure",
                f"{withdrawal_result['withdrawal_rate']:.1%}"
            )

            st.write(
                f"LCR at threshold: "
                f"**{withdrawal_result['lcr']:.1f}%**"
            )

        else:

            st.success(
                "LCR remained above 100% across the tested withdrawal range."
            )

    with col2:

        st.markdown("#### Interest Rate Threshold")

        if interest_result["threshold_found"]:

            st.metric(
                "Critical Rate Shock",
                f"+{interest_result['interest_rate_shock']:.2f} pp"
            )

            st.write(
                f"LCR at threshold: "
                f"**{interest_result['lcr']:.1f}%**"
            )

        else:

            st.success(
                "LCR remained above 100% across the tested rate-shock range."
            )


# ============================================================
# RECOVERY SIMULATOR
# ============================================================

with recovery_tab:

    st.markdown('<div class="section-title">🛡️ Recovery Simulator</div>', unsafe_allow_html=True)

    st.write(
        "Test how additional HQLA could restore liquidity to the "
        "100% project/reference threshold."
    )

    recovery_source = st.selectbox(
        "Recovery scenario",
        [
            "Current baseline",
            "Worst simulated scenario"
        ]
    )

    if recovery_source == "Current baseline":

        current_hqla = hqla
        recovery_outflow = cash_outflow
        recovery_inflow = cash_inflow

    else:

        worst = (
            df[
                df["lcr"].notna()
            ]
            .sort_values("lcr")
            .iloc[0]
        )

        current_hqla = worst["hqla"]
        recovery_outflow = worst["cash_outflow"]
        recovery_inflow = worst["cash_inflow"]

    recovery = calculate_additional_hqla(
        current_hqla=current_hqla,
        cash_outflow=recovery_outflow,
        cash_inflow=recovery_inflow,
        target_lcr=100
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Current HQLA",
        f"₹{recovery['current_hqla']:,.1f} Cr"
    )

    col2.metric(
        "Required HQLA",
        f"₹{recovery['required_hqla']:,.1f} Cr"
    )

    col3.metric(
        "Additional HQLA Needed",
        f"₹{recovery['additional_hqla']:,.1f} Cr"
    )

    st.markdown("### What-if Recovery")

    additional_hqla = st.slider(
        "Additional HQLA (₹ Crore)",
        min_value=0.0,
        max_value=float(
            max(recovery["additional_hqla"] * 1.5, 100)
        ),
        value=0.0,
        step=10.0
    )

    projected = simulate_recovery(
        current_hqla=current_hqla,
        cash_outflow=recovery_outflow,
        cash_inflow=recovery_inflow,
        additional_hqla=additional_hqla
    )

    st.metric(
        "Projected LCR",
        f"{projected['projected_lcr']:.1f}%"
    )

    if projected["projected_lcr"] >= 100:

        st.success(
            "The selected recovery action brings LCR to or above "
            "the project/reference threshold."
        )

    else:

        remaining = (
            recovery["required_hqla"]
            - projected["projected_hqla"]
        )

        st.warning(
            f"Approximately ₹{max(remaining, 0):,.1f} Cr additional HQLA "
            "would still be required."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <b>Liquidity Risk Decision Platform</b> · Monte Carlo · DuckDB · Streamlit · Plotly<br>
        All scenarios and stress assumptions are synthetic and intended for simulation/demo purposes.
    </div>
    """,
    unsafe_allow_html=True
)