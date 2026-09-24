
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

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
# HISTORICAL ML EARLY WARNING
# ============================================================

HISTORICAL_FILE = Path("data/ml_ready_bank_liquidity.csv")
MODEL_FILE = Path("ml/liquidity_risk_model.joblib")


@st.cache_data
def load_historical_data():
    if not HISTORICAL_FILE.exists():
        return None
    data = pd.read_csv(HISTORICAL_FILE)
    data["REPORTINGDATE"] = pd.to_datetime(data["REPORTINGDATE"], errors="coerce")
    return data.dropna(subset=["REPORTINGDATE"]).sort_values(
        ["INSTITUTIONCODE", "REPORTINGDATE"]
    )


@st.cache_resource
def load_risk_model():
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Liquidity Risk Simulator",
    page_icon="📊",
    layout="wide"
)

# Load historical ML data only AFTER set_page_config().
historical_df = load_historical_data()
risk_model_bundle = load_risk_model()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
:root {
    --bg:#F4F7F6;
    --surface:#FFFFFF;
    --surface-soft:#F7FBFA;
    --teal-deep:#003135;
    --teal:#0FA4AF;
    --teal-mid:#147B83;
    --aqua:#AFDDE5;
    --aqua-soft:#E7F5F6;
    --rust:#964734;
    --rust-soft:#F6E8E3;
    --cream:#F5F1EA;
    --ink:#17383A;
    --muted:#607779;
    --line:#D7E4E4;
    --shadow:rgba(0,49,53,.10);
}

/* ---------- Global ---------- */
.stApp {
    background:linear-gradient(180deg,#F8FAF9 0%,#F0F5F4 100%);
    color:var(--ink);
    font-size:15px;
}
[data-testid="stAppViewContainer"] { background:#F4F7F6; }
[data-testid="stHeader"] { background:rgba(244,247,246,.94); }
.block-container {
    padding-top:1.35rem;
    padding-bottom:2rem;
    max-width:1500px;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#003135 0%,#07474C 62%,#0B555B 100%);
    border-right:1px solid rgba(175,221,229,.16);
}
[data-testid="stSidebar"] * { color:#F5FCFC !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stMarkdown p {
    color:#D8ECEE !important;
    font-size:.92rem !important;
}
[data-testid="stSidebar"] .stMarkdown {
    font-size:.92rem;
}

/* Inputs */
[data-testid="stSidebar"] input {
    background:#FFFFFF !important;
    border:1px solid #C9DCDD !important;
    color:#17383A !important;
    -webkit-text-fill-color:#17383A !important;
    border-radius:10px !important;
    font-size:.95rem !important;
    font-weight:600 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background:#FFFFFF !important;
    border:1px solid #C9DCDD !important;
    color:#17383A !important;
    border-radius:10px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] div[role="button"] {
    color:#17383A !important;
    -webkit-text-fill-color:#17383A !important;
}

/* Main sidebar action */
[data-testid="stSidebar"] [data-testid="stButton"] button,
[data-testid="stSidebar"] button[kind="primary"] {
    background:linear-gradient(135deg,#AFDDE5,#8CCFD6) !important;
    border:1px solid #79BEC6 !important;
    color:#003135 !important;
    -webkit-text-fill-color:#003135 !important;
    font-weight:850 !important;
    font-size:.94rem !important;
    border-radius:11px !important;
    box-shadow:0 4px 0 #5B9DA5,0 9px 20px rgba(0,49,53,.20) !important;
    text-shadow:none !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button *,
[data-testid="stSidebar"] button[kind="primary"] * {
    background:transparent !important;
    color:#003135 !important;
    -webkit-text-fill-color:#003135 !important;
    font-weight:850 !important;
    text-shadow:none !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover,
[data-testid="stSidebar"] button[kind="primary"]:hover {
    background:#C9E9ED !important;
    color:#003135 !important;
    transform:translateY(-1px);
}
[data-testid="stSidebar"] [data-testid="stButton"] button:active,
[data-testid="stSidebar"] button[kind="primary"]:active {
    transform:translateY(1px);
    box-shadow:0 2px 0 #5B9DA5,0 5px 10px rgba(0,49,53,.15) !important;
}

/* Sidebar number controls */
[data-testid="stSidebar"] [data-testid="stNumberInput"] > div {
    border-radius:10px !important;
    overflow:hidden !important;
    border:1px solid #C9DCDD !important;
    background:#FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    background:#003135 !important;
    color:#FFFFFF !important;
    -webkit-text-fill-color:#FFFFFF !important;
    border:0 !important;
    border-left:1px solid rgba(255,255,255,.18) !important;
    border-radius:0 !important;
    min-width:35px !important;
    min-height:40px !important;
    height:40px !important;
    font-size:18px !important;
    font-weight:850 !important;
    opacity:1 !important;
    box-shadow:none !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button:hover {
    background:#0F5960 !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button svg,
[data-testid="stSidebar"] [data-testid="stNumberInput"] button svg path,
[data-testid="stSidebar"] [data-testid="stNumberInput"] button svg line,
[data-testid="stSidebar"] [data-testid="stNumberInput"] button svg polyline {
    width:16px !important;
    height:16px !important;
    color:#FFFFFF !important;
    fill:#FFFFFF !important;
    stroke:#FFFFFF !important;
    opacity:1 !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    background:#FFFFFF !important;
    color:#17383A !important;
    -webkit-text-fill-color:#17383A !important;
    font-weight:650 !important;
    border:0 !important;
    font-size:.96rem !important;
}

/* ---------- Sidebar slider: clean visible track ---------- */
[data-testid="stSidebar"] [data-testid="stSlider"] {
    padding-top:6px !important;
    padding-bottom:5px !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] {
    padding:7px 2px 8px !important;
}
/* BaseWeb slider track */
[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
    background:transparent !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child > div {
    background:#76AEB3 !important;
    height:5px !important;
    border-radius:999px !important;
}
/* Filled portion + thumb */
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
    background:#AFDDE5 !important;
    border:3px solid #003135 !important;
    box-shadow:0 2px 7px rgba(0,49,53,.30) !important;
    width:16px !important;
    height:16px !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBar"] {
    background:transparent !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color:#D8ECEE !important;
}

/* ---------- Hero ---------- */
.hero-card {
    background:
        linear-gradient(135deg,#003135 0%,#07525A 56%,#147B83 100%);
    border-radius:20px;
    padding:30px 34px;
    margin:4px 0 20px;
    color:white;
    box-shadow:0 12px 30px rgba(0,49,53,.18);
    position:relative;
    overflow:hidden;
    border:1px solid rgba(175,221,229,.20);
    min-height:126px;
}

/* Subtle financial/network pattern on the right */
.hero-card:before {
    content:"";
    position:absolute;
    inset:0;
    right:0;
    width:48%;
    background:
        radial-gradient(circle at 72% 34%,rgba(175,221,229,.65) 0 2px,transparent 3px),
        radial-gradient(circle at 86% 22%,rgba(175,221,229,.55) 0 2px,transparent 3px),
        radial-gradient(circle at 91% 55%,rgba(175,221,229,.50) 0 2px,transparent 3px),
        radial-gradient(circle at 66% 70%,rgba(175,221,229,.48) 0 2px,transparent 3px),
        linear-gradient(24deg,transparent 48%,rgba(175,221,229,.13) 49%,rgba(175,221,229,.13) 50%,transparent 51%),
        linear-gradient(154deg,transparent 48%,rgba(175,221,229,.12) 49%,rgba(175,221,229,.12) 50%,transparent 51%),
        linear-gradient(75deg,transparent 48%,rgba(175,221,229,.10) 49%,rgba(175,221,229,.10) 50%,transparent 51%);
    background-size:
        100% 100%,100% 100%,100% 100%,100% 100%,
        130px 95px,150px 110px,170px 120px;
    opacity:.72;
    pointer-events:none;
    animation:heroNetwork 7s ease-in-out infinite;
}

.hero-card:after {
    content:"";
    position:absolute;
    width:300px;
    height:300px;
    right:-110px;
    top:-145px;
    border-radius:50%;
    background:rgba(175,221,229,.13);
    box-shadow:0 0 0 35px rgba(175,221,229,.055);
    pointer-events:none;
}

@keyframes heroNetwork {
    0%,100% { opacity:.58; transform:translateX(0); }
    50% { opacity:.78; transform:translateX(-5px); }
}

.hero-content {
    position:relative;
    z-index:2;
    max-width:980px;
}

.hero-badge {
    display:inline-flex;
    align-items:center;
    gap:7px;
    padding:5px 10px;
    margin-bottom:9px;
    border-radius:999px;
    background:rgba(175,221,229,.14);
    border:1px solid rgba(175,221,229,.30);
    color:#DDF6F8;
    font-size:.68rem;
    font-weight:850;
    letter-spacing:.12em;
    text-transform:uppercase;
}

.hero-badge-dot {
    width:7px;
    height:7px;
    border-radius:50%;
    background:#AFDDE5;
    box-shadow:0 0 0 4px rgba(175,221,229,.10);
    animation:heroPulse 1.8s ease-in-out infinite;
}

@keyframes heroPulse {
    0%,100% { opacity:.55; transform:scale(.9); }
    50% { opacity:1; transform:scale(1.1); }
}

.hero-title {
    font-size:2rem;
    line-height:1.15;
    font-weight:850;
    margin:0;
    color:#FFFFFF;
}

.hero-subtitle {
    color:#E5F3F4;
    font-size:1rem;
    line-height:1.6;
    max-width:980px;
    margin:9px 0 0;
}

/* ---------- Typography ---------- */
.section-title {
    color:var(--teal-deep);
    font-size:1.48rem;
    font-weight:850;
    margin:9px 0 5px;
    display:flex;
    align-items:center;
    gap:10px;
}
.section-title:before {
    content:"";
    width:5px;
    height:25px;
    border-radius:4px;
    background:linear-gradient(180deg,#0FA4AF,#003135);
    display:inline-block;
    box-shadow:1px 2px 5px rgba(0,49,53,.18);
}
.section-subtitle {
    color:var(--muted);
    margin-bottom:18px;
    font-size:.96rem;
}

/* ---------- Metric cards ---------- */
.metric-card {
    background:linear-gradient(180deg,#FFFFFF 0%,#FBFDFC 100%);
    border:1px solid var(--line);
    border-radius:15px;
    padding:17px 18px;
    box-shadow:0 5px 0 #DCE8E8,0 11px 25px rgba(0,49,53,.07);
    min-height:112px;
    position:relative;
    overflow:hidden;
    transition:transform .18s ease,box-shadow .18s ease,border-color .18s ease;
}
.metric-card:hover {
    transform:translateY(-3px);
    border-color:#A9CED1;
    box-shadow:0 7px 0 #CFE0E1,0 16px 30px rgba(0,49,53,.12);
}
.metric-top {
    display:flex;
    align-items:center;
    gap:9px;
}
.metric-icon {
    width:34px;
    height:34px;
    border-radius:10px;
    display:flex;
    align-items:center;
    justify-content:center;
    background:linear-gradient(145deg,#DDF3F5,#AFDDE5);
    color:#003135;
    font-size:17px;
    font-weight:900;
    box-shadow:inset 0 1px 0 rgba(255,255,255,.9),0 3px 8px rgba(0,49,53,.12);
}
.metric-label {
    color:#526B6D;
    font-size:.82rem;
    font-weight:800;
    text-transform:uppercase;
    letter-spacing:.045em;
}
.metric-value {
    color:#003135;
    font-size:1.62rem;
    font-weight:850;
    margin-top:7px;
}
.metric-note {
    color:#74898A;
    font-size:.78rem;
    margin-top:4px;
}

/* ---------- Risk banners ---------- */
.risk-safe,.risk-warning,.risk-danger {
    padding:15px 19px;
    border-radius:12px;
    margin:15px 0 21px;
    font-size:.94rem;
}
.risk-safe {
    background:#EDF7F4;
    border:1px solid #C8E1D9;
    border-left:5px solid #278A73;
    color:#236454;
}
.risk-warning {
    background:#FFF4E9;
    border:1px solid #EBC8A9;
    border-left:5px solid #B56A3F;
    color:#7B492F;
}
.risk-danger {
    background:#F9ECE8;
    border:1px solid #E2BBB0;
    border-left:5px solid #964734;
    color:#713D32;
}

/* ---------- Tabs: complementary aqua/teal ---------- */
[data-baseweb="tab-list"] {
    gap:6px !important;
    background:#E7F5F6 !important;
    border:1px solid #C9E2E4 !important;
    border-radius:13px !important;
    padding:5px !important;
}
button[data-baseweb="tab"] {
    font-weight:750 !important;
    font-size:.90rem !important;
    color:#4E6C6F !important;
    padding:9px 13px !important;
    border-radius:9px !important;
    background:transparent !important;
    transition:all .18s ease;
}
button[data-baseweb="tab"]:hover {
    color:#003135 !important;
    background:#D5EEF0 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color:#FFFFFF !important;
    background:#003135 !important;
    box-shadow:0 3px 9px rgba(0,49,53,.18);
}
button[data-baseweb="tab"][aria-selected="true"] * {
    color:#FFFFFF !important;
}
div[data-baseweb="tab-highlight"] {
    background:transparent !important;
    height:0 !important;
}

/* ---------- Persistent navigation control ---------- */

div[data-testid="stElementContainer"]:has([data-testid="stSegmentedControl"]) {
    width:100% !important;
    max-width:none !important;
    min-width:100% !important;
}

div[data-testid="stElementContainer"]:has([data-testid="stSegmentedControl"]) > div {
    width:100% !important;
    max-width:none !important;
    min-width:100% !important;
}

[data-testid="stSegmentedControl"] {
    width:100% !important;
    max-width:none !important;
    min-width:100% !important;
    box-sizing:border-box !important;

    background:linear-gradient(135deg,#F4FBFB 0%,#E7F5F6 100%) !important;
    border:1px solid #BFDADC !important;
    border-radius:18px !important;
    padding:8px !important;
    margin:5px 0 26px !important;

    box-shadow:0 7px 20px rgba(0,49,53,.10) !important;
}


/* THIS is the important part — force the inner BaseWeb group full width */
[data-testid="stSegmentedControl"] [data-baseweb="button-group"] {
    width:100% !important;
    max-width:none !important;
    min-width:100% !important;

    display:grid !important;
    grid-template-columns:repeat(7, minmax(0, 1fr)) !important;
    gap:8px !important;
}


/* Individual navigation buttons */
[data-testid="stSegmentedControl"] [data-baseweb="button-group"] > button {
    width:100% !important;
    min-width:0 !important;
    min-height:70px !important;

    padding:16px 18px !important;

    color:#466568 !important;
    background:transparent !important;

    border:1px solid transparent !important;
    border-radius:13px !important;

    font-weight:800 !important;
    font-size:1.12rem !important;
    line-height:1.25 !important;

    white-space:nowrap !important;
    box-sizing:border-box !important;

    transition:all .18s ease !important;
}


/* Hover */
[data-testid="stSegmentedControl"] [data-baseweb="button-group"] > button:hover {
    color:#003135 !important;
    background:#D8F0F2 !important;
    border-color:#B8DDE0 !important;

    transform:translateY(-1px);
    box-shadow:0 3px 8px rgba(0,49,53,.08) !important;
}


/* Active tab */
[data-testid="stSegmentedControl"] [data-baseweb="button-group"] > button[aria-pressed="true"] {
    color:#FFFFFF !important;

    background:linear-gradient(
        135deg,
        #003135 0%,
        #0B5B61 100%
    ) !important;

    border-color:#003135 !important;

    box-shadow:0 5px 13px rgba(0,49,53,.25) !important;

    transform:translateY(-1px);
}

[data-testid="stSegmentedControl"] [data-baseweb="button-group"] > button[aria-pressed="true"] * {
    color:#FFFFFF !important;
}

/* ---------- Main buttons ---------- */
.stButton > button,.stDownloadButton > button {
    border-radius:10px !important;
    font-weight:750 !important;
    font-size:.90rem !important;
    border:1px solid #BFD6D7 !important;
    background:#FFFFFF !important;
    color:#003135 !important;
}
.stButton > button:hover,.stDownloadButton > button:hover {
    border-color:#0FA4AF !important;
    color:#003135 !important;
    box-shadow:0 5px 15px rgba(0,49,53,.10);
}

/* ---------- Inputs / tables ---------- */
[data-baseweb="select"] > div,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    border-radius:9px !important;
    border-color:#C9DCDD !important;
    font-size:.93rem !important;
}
[data-testid="stDataFrame"] {
    border:1px solid var(--line);
    border-radius:12px;
    overflow:hidden;
}
[data-testid="stExpander"] {
    border:1px solid var(--line);
    border-radius:12px;
    background:#FFFFFF;
}

/* Quick guide + sidebar helper cards */
.quick-icon {
    display:inline-flex;
    width:28px;
    height:28px;
    align-items:center;
    justify-content:center;
    border-radius:8px;
    background:rgba(175,221,229,.18);
    margin-right:7px;
}
[data-testid="stSidebar"] .stMarkdown b {
    color:#FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Simulation Controls")

st.sidebar.markdown(
    "Set the bank assumptions and run the stress test."
)
st.sidebar.markdown(
    '<div style="font-size:.78rem;font-weight:800;letter-spacing:.08em;'
    'text-transform:uppercase;color:#AFDDE5 !important;margin:18px 0 9px;font-size:.82rem;">'
    '◈&nbsp;&nbsp;Baseline Parameters</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div style="background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12);
    border-radius:12px;padding:12px 13px;margin:14px 0 18px;">
      <div style="font-weight:800;color:#FFFFFF;margin-bottom:6px;"><span class="quick-icon">✦</span>Quick Guide</div>
      <div style="font-size:.84rem;line-height:1.60;color:#D8ECEE;">
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
    "💧 HQLA (₹ Crore)",
    min_value=100.0,
    value=1200.0,
    step=50.0
)

cash_outflow = st.sidebar.number_input(
    "↗ Base Cash Outflow (₹ Crore)",
    min_value=100.0,
    value=1000.0,
    step=50.0
)

cash_inflow = st.sidebar.number_input(
    "↘ Base Cash Inflow (₹ Crore)",
    min_value=0.0,
    value=250.0,
    step=25.0
)

base_interest_rate = st.sidebar.number_input(
    "％ Base Interest Rate (%)",
    min_value=0.0,
    value=6.5,
    step=0.25
)
base_withdrawal_percent = st.sidebar.slider(
    "⇩ Base Withdrawal Pressure",
    min_value=0,
    max_value=35,
    value=5,
    step=1,
    format="%d%%"
)

base_withdrawal_rate = base_withdrawal_percent / 100

st.sidebar.markdown(
    '<div style="font-size:.80rem;color:#CFE3E5 !important;margin:-2px 0 10px;">'
    'These values define the starting point for the Monte Carlo stress simulation.'
    '</div>',
    unsafe_allow_html=True
)

n_simulations = st.sidebar.selectbox(
    "◎ Number of Scenarios",
    [1000, 5000, 10000],
    index=2
)

run_button = st.sidebar.button(
    "▶  Run Stress Simulation",
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
        font-size:.82rem;
        line-height:1.45;
    ">
        <b style="color:#FFFFFF;">💡 Tip:</b>
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
        <div class="hero-content">
            <div class="hero-badge">
                <span class="hero-badge-dot"></span>
                Real-Time Stress Engine
            </div>
            <div class="hero-title">◉ Liquidity Risk Simulator</div>
            <p class="hero-subtitle">
                Explore how withdrawal pressure and interest-rate shocks affect bank liquidity.
                Run Monte Carlo scenarios, review LCR, identify risk drivers, and test recovery actions.
            </p>
        </div>
    </div>
    """,
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
    ("◌", "Scenarios", f"{total_scenarios:,}", "Generated futures"),
    ("↗", "Average LCR", f"{average_lcr:.1f}%", "Portfolio-level average"),
    ("▥", "Median LCR", f"{median_lcr:.1f}%", "Middle simulated outcome"),
    ("⌁", "Worst LCR", f"{worst_lcr:.1f}%", "Lowest simulated outcome"),
    ("!", "Below 100%", f"{percentage_below:.1f}%", "Scenarios under threshold"),
]

for col, (icon, label, value, note) in zip(metric_cols, metric_items):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-top">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-label">{label}</div>
                </div>
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
        <span style="font-size:1.05rem;margin-right:5px;">◉</span><b>Liquidity Status:</b> Most simulated scenarios remain above
        the project/reference threshold.
        </div>
        """,
        unsafe_allow_html=True
    )

elif average_lcr >= 100:

    st.markdown(
        """
        <div class="risk-warning">
        <span style="font-size:1.05rem;margin-right:5px;">◉</span><b>Liquidity Status:</b> Average simulated LCR remains above the
        reference threshold, but a meaningful subset of scenarios falls below it.
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
        <div class="risk-danger">
        <span style="font-size:1.05rem;margin-right:5px;">◉</span><b>Liquidity Status:</b> The simulated environment produces an
        average LCR below the project/reference threshold.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TABS
# ============================================================

TAB_OPTIONS = [
    "◉ Overview",
    "△ Stress Lab",
    "▦ Scenarios",
    "◎ Early Warning",
    "◇ Risk Diagnosis",
    "◆ Reverse Stress",
    "↗ Recovery"
]

TAB_PANEL_KEYS = {
    "◉ Overview": "panel_overview",
    "△ Stress Lab": "panel_stress",
    "▦ Scenarios": "panel_scenarios",
    "◎ Early Warning": "panel_early_warning",
    "◇ Risk Diagnosis": "panel_diagnosis",
    "◆ Reverse Stress": "panel_reverse_stress",
    "↗ Recovery": "panel_recovery",
}

active_tab = st.segmented_control(
    "Navigation",
    TAB_OPTIONS,
    default=TAB_OPTIONS[0],
    key="active_tab",
    selection_mode="single",
    label_visibility="collapsed"
)

if active_tab is None:
    active_tab = TAB_OPTIONS[0]

# Keep every panel mounted so all widget/session states survive reruns.
# Only the selected panel is visually shown.
hidden_panels = [
    panel_key
    for tab_name, panel_key in TAB_PANEL_KEYS.items()
    if tab_name != active_tab
]

hide_css = "".join(
    f".st-key-{panel_key}{{display:none !important;}}"
    for panel_key in hidden_panels
)

st.markdown(
    f"<style>{hide_css}</style>",
    unsafe_allow_html=True
)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

with st.container(key="panel_overview"):

    st.markdown(
        '<div class="section-title">Liquidity Distribution</div>',
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

    st.markdown("### ◇ Stress Relationships")

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

with st.container(key="panel_stress"):

    st.markdown('<div class="section-title">Stress Lab</div>', unsafe_allow_html=True)

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

    st.markdown("### ▣ Model Inputs")

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

with st.container(key="panel_scenarios"):

    st.markdown('<div class="section-title">Scenario Explorer</div>', unsafe_allow_html=True)

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
            "Search Scenario ID",
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
        label=f"Download All {len(df):,} Scenarios (CSV)",
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

    st.markdown("### ⚠ Worst 10 Scenarios")

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
# ML EARLY WARNING
# ============================================================

with st.container(key="panel_early_warning"):

    st.markdown(
        '<div class="section-title">Historical ML Early Warning</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-subtitle">Use past bank observations to estimate the next-period liquidity risk band.</div>',
        unsafe_allow_html=True
    )

    if historical_df is None or risk_model_bundle is None:
        st.error(
            "Early Warning is unavailable. Make sure both "
            "data/ml_ready_bank_liquidity.csv and "
            "ml/liquidity_risk_model.joblib exist."
        )
    else:
        institutions = sorted(
            historical_df["INSTITUTIONCODE"].dropna().astype(str).unique()
        )

        selected_institution = st.selectbox(
            "Institution Code",
            institutions,
            key="early_warning_institution"
        )

        institution_df = historical_df[
            historical_df["INSTITUTIONCODE"].astype(str) == selected_institution
        ].sort_values("REPORTINGDATE")

        latest = institution_df.iloc[-1]

        model = risk_model_bundle["model"]
        model_features = risk_model_bundle["features"]
        risk_bands = risk_model_bundle.get(
            "risk_bands",
            {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
        )

        missing_features = [
            feature for feature in model_features
            if feature not in latest.index
        ]

        if missing_features:
            st.error(
                "The saved model expects features that are missing from the "
                f"historical dataset: {', '.join(missing_features)}"
            )
        else:
            X_latest = pd.DataFrame(
                [[latest[feature] for feature in model_features]],
                columns=model_features
            )

            prediction = int(model.predict(X_latest)[0])
            predicted_label = risk_bands.get(prediction, str(prediction))

            risk_class = {
                0: "risk-safe",
                1: "risk-warning",
                2: "risk-danger"
            }.get(prediction, "risk-warning")

            st.markdown(
                f'''<div class="{risk_class}">
                <b>Predicted Next-Period Risk:</b> {predicted_label}<br>
                <span style="font-size:.82rem;">Random Forest · Historical observations</span>
                </div>''',
                unsafe_allow_html=True
            )

            st.caption(
                f"Latest observation: {latest['REPORTINGDATE'].date()} • "
                f"Institution: {selected_institution}"
            )

            metric_cols = st.columns(4)

            liquidity_proxy = latest.get("LIQUIDITY_PROXY")
            loan_to_deposit = latest.get("LOAN_TO_DEPOSIT")
            npl_ratio = latest.get("NPL_RATIO")
            cash_liquidity = latest.get("CASH_TO_LIQUID_LIAB")

            metrics = [
                ("Liquidity Proxy", liquidity_proxy, "Liquid assets / liquid liabilities"),
                ("Loan / Deposit", loan_to_deposit, "Credit funding pressure"),
                ("NPL Ratio", npl_ratio, "Non-performing loan indicator"),
                ("Cash / Liquid Liab.", cash_liquidity, "Cash liquidity position")
            ]

            for col, (label, value, note) in zip(metric_cols, metrics):
                with col:
                    if pd.isna(value):
                        display_value = "—"
                    else:
                        display_value = f"{value:.3f}"
                    st.markdown(
                        f'''<div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{display_value}</div>
                        <div class="metric-note">{note}</div>
                        </div>''',
                        unsafe_allow_html=True
                    )

            st.markdown("### ◈ Model Risk Drivers")
            st.caption(
                "These are the most influential features in the trained model; "
                "they do not explain causation for one institution."
            )

            importance_df = pd.DataFrame({
                "Feature": model_features,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False).head(10)

            importance_df["Importance"] = (
                importance_df["Importance"] * 100
            ).round(2)
            importance_df = importance_df.rename(
                columns={"Importance": "Importance (%)"}
            )

            st.dataframe(
                importance_df,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### ⌁ Historical Trends")

            trend_columns = [
                "LIQUIDITY_PROXY",
                "LOAN_TO_DEPOSIT",
                "NPL_RATIO"
            ]
            available_trends = [
                col for col in trend_columns if col in institution_df.columns
            ]

            if available_trends:
                trend_df = institution_df[
                    ["REPORTINGDATE"] + available_trends
                ].copy()
                trend_df = trend_df.set_index("REPORTINGDATE")
                trend_df = trend_df.rename(columns={
                    "LIQUIDITY_PROXY": "Liquidity Proxy",
                    "LOAN_TO_DEPOSIT": "Loan / Deposit",
                    "NPL_RATIO": "NPL Ratio"
                })

                st.line_chart(trend_df, use_container_width=True)

# ============================================================
# RISK DIAGNOSIS
# ============================================================

with st.container(key="panel_diagnosis"):

    st.markdown('<div class="section-title">Why Did LCR Fall?</div>', unsafe_allow_html=True)

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

with st.container(key="panel_reverse_stress"):

    st.markdown('<div class="section-title">Reverse Stress Testing</div>', unsafe_allow_html=True)

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

with st.container(key="panel_recovery"):

    st.markdown('<div class="section-title">Recovery Simulator</div>', unsafe_allow_html=True)

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

    st.markdown("### ↗ What-if Recovery")

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



