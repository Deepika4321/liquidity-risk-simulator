
@"
# 💧 Real-Time Financial Liquidity Risk & Stress-Testing Simulator

> **A quantitative liquidity risk decision platform that simulates thousands of stressed financial futures, measures Liquidity Coverage Ratio (LCR), identifies risk drivers, discovers critical stress thresholds, and evaluates recovery actions.**

---

## 🚨 Problem Statement

Financial institutions need to understand how their liquidity position may behave under uncertain and stressful conditions.

A single financial forecast is not enough because liquidity can deteriorate due to multiple factors such as:

- Increased customer withdrawals
- Higher interest-rate shocks
- Rising cash outflows
- Declining cash inflows
- Reduction in the value of High-Quality Liquid Assets (HQLA)

The challenge is to simulate these possible situations and determine **how frequently the institution's liquidity position falls below a critical threshold**.

---

## 💡 Our Solution

The **Real-Time Financial Liquidity Risk & Stress-Testing Simulator** uses a **Monte Carlo simulation engine** to generate thousands of synthetic financial scenarios.

For every scenario, the system:

1. Generates a financial stress environment.
2. Applies interest-rate and withdrawal shocks.
3. Adjusts cash outflows and inflows.
4. Applies an HQLA stress haircut.
5. Calculates Net Cash Outflow.
6. Calculates the Liquidity Coverage Ratio (LCR).
7. Classifies the scenario by stress severity.
8. Stores the generated scenarios in DuckDB.
9. Visualizes the resulting liquidity risk.
10. Identifies critical stress thresholds and possible recovery actions.

The platform is designed as an **interactive decision-support tool**, rather than a static financial dashboard.

---

# 🎯 Key Features

## 1. 🎲 Monte Carlo Stress Simulation

The core engine generates up to **10,000 synthetic liquidity scenarios**.

Each scenario contains variables such as:

- Interest-rate shock
- Effective interest rate
- Withdrawal pressure
- Stressed HQLA
- Stressed cash outflow
- Stressed cash inflow
- Net cash outflow
- LCR
- Stress score
- Stress category

The simulation uses a fixed random seed for reproducibility.

---

## 2. 📊 Liquidity Coverage Ratio (LCR)

The simulator calculates LCR using:

**LCR = HQLA / Net Cash Outflow × 100**

where:

**Net Cash Outflow = Cash Outflow − Cash Inflow**

The dashboard tracks:

- Average LCR
- Median LCR
- Worst simulated LCR
- Percentage of scenarios below 100%
- Number of generated scenarios

For this hackathon project, **100% is used as the project/reference threshold**.

---

## 3. 🔥 Stress Lab

The Stress Lab provides a view of how liquidity behaves as financial stress increases.

It analyzes relationships between:

- Stress score vs LCR
- Interest-rate shock vs LCR
- Withdrawal pressure vs LCR

It also categorizes scenarios into:

| Stress Category | Stress Score |
|---|---:|
| Normal | 0 – <0.25 |
| Moderate | 0.25 – <0.50 |
| Severe | 0.50 – <0.75 |
| Extreme | 0.75 – 1.00 |

These categories are **project-defined analytical categories** for the simulation.

---

## 4. 🔎 Scenario Explorer

Users can investigate individual simulated futures instead of looking only at aggregate metrics.

The Scenario Explorer supports:

- Stress-category filtering
- LCR filtering
- Scenario ID search
- Pagination
- Configurable rows per page
- Sorting by LCR
- Worst 10 scenario analysis
- Complete scenario CSV download

This makes the simulator useful for investigating specific high-risk scenarios.

---

## 5. 🧠 Risk Diagnosis

The Risk Diagnosis module answers:

> **"Why did LCR fall?"**

The system analyzes the worst simulated scenario and compares it against the configured baseline.

It examines changes in:

- HQLA
- Cash outflow
- Cash inflow
- Interest rate
- LCR

The module identifies the primary simulated driver and presents the relative changes in a transparent table.

The attribution is based on the project's simulation model and is **not an empirical banking attribution model**.

---

## 6. 🎯 Reverse Stress Testing

Instead of asking:

> "What happens under this stress?"

the simulator can ask:

> **"How much stress is required for LCR to fall below 100%?"**

The Reverse Stress module searches for approximate critical thresholds for:

### Withdrawal Pressure

Determines the withdrawal pressure at which LCR reaches the reference threshold.

### Interest-Rate Shock

Determines the approximate interest-rate shock at which LCR reaches the reference threshold.

This provides a more decision-oriented view of liquidity risk.

---

## 7. 🛡️ Recovery Simulator

The Recovery Simulator evaluates how additional HQLA can restore liquidity.

Users can choose between:

- Current baseline
- Worst simulated scenario

The system calculates:

- Current HQLA
- Required HQLA
- Additional HQLA needed
- Projected LCR after recovery

Users can then interactively test different additional-HQLA amounts and immediately see the projected LCR.

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────────┐
                    │       Streamlit UI      │
                    │                         │
                    │ Inputs • Charts • Tabs  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Risk Engine Layer    │
                    │                         │
                    │ LCR • Stress • Recovery │
                    │ Reverse Stress • Risk   │
                    │ Diagnosis               │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Monte Carlo Engine    │
                    │                         │
                    │ Thousands of synthetic  │
                    │ financial futures       │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │        DuckDB           │
                    │                         │
                    │ Scenario storage &      │
                    │ analytical queries      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      Plotly Charts      │
                    │                         │
                    │ Distribution • Stress   │
                    │ Relationships • Trends  │
                    └─────────────────────────┘