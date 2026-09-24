import pandas as pd
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

RAW_FILE = Path("data/BankLiquidityRiskDetection.csv")
CLEAN_FILE = Path("data/cleaned_bank_liquidity.csv")
ML_OUTPUT = Path("data/ml_ready_bank_liquidity.csv")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(RAW_FILE)

print("Raw shape:", df.shape)


# ============================================================
# SELECT RELEVANT FEATURES
# ============================================================

SELECTED_COLUMNS = [
    "REPORTINGDATE",
    "INSTITUTIONCODE",

    # Liquidity
    "13_CASH",
    "22_TREASURY_BILLS",
    "23_OTHER_GOV_SECURITIES",
    "XX_CUSTOMER_DEPOSITS",
    "XX_TOTAL_LIQUID_LIAB",
    "XX_TOTAL_LIQUID_ASSET",
    "XX_MLA",

    # Balance sheet
    "F077_ASSETS_TOTAL",
    "F125_LIAB_TOTAL",

    # Credit risk
    "EWAQ_GrossLoans",
    "EWAQ_Capital",
    "EWAQ_NPL",
    "EWAQ_NPLsNetOfProvisions",
    "EWAQ_NPLsNetOfProvisions2CoreCapital",

    # Existing liquidity/risk indicators
    "LR",
    "DR",
    "IBCM",

    # Macro-economic
    "GDP",
    "INF",

    # Current liquidity rating
    "EWL_LIQUIDITY RATING"
]

df = df[SELECTED_COLUMNS].copy()


# ============================================================
# DATE CLEANING
# ============================================================

df["REPORTINGDATE"] = pd.to_datetime(
    df["REPORTINGDATE"],
    errors="coerce"
)

df = df.dropna(
    subset=["REPORTINGDATE"]
)


# ============================================================
# SORT BY INSTITUTION + DATE
# ============================================================

df = df.sort_values(
    ["INSTITUTIONCODE", "REPORTINGDATE"]
).reset_index(drop=True)


# ============================================================
# REMOVE DUPLICATE INSTITUTION-DATE ROWS
# ============================================================

before = len(df)

df = df.drop_duplicates(
    subset=[
        "INSTITUTIONCODE",
        "REPORTINGDATE"
    ]
)

after = len(df)

print(
    "Duplicate institution-date rows removed:",
    before - after
)


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    column
    for column in SELECTED_COLUMNS
    if column not in [
        "REPORTINGDATE",
        "INSTITUTIONCODE"
    ]
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

# 1. Liquidity Proxy
df["LIQUIDITY_PROXY"] = (
    df["XX_TOTAL_LIQUID_ASSET"]
    / df["XX_TOTAL_LIQUID_LIAB"].replace(
        0,
        pd.NA
    )
)


# 2. Cash to Liquid Liabilities
df["CASH_TO_LIQUID_LIAB"] = (
    df["13_CASH"]
    / df["XX_TOTAL_LIQUID_LIAB"].replace(
        0,
        pd.NA
    )
)


# 3. Liquid Assets to Total Assets
df["LIQUID_ASSET_TO_TOTAL_ASSET"] = (
    df["XX_TOTAL_LIQUID_ASSET"]
    / df["F077_ASSETS_TOTAL"].replace(
        0,
        pd.NA
    )
)


# 4. Loan to Deposit Ratio
df["LOAN_TO_DEPOSIT"] = (
    df["EWAQ_GrossLoans"]
    / df["XX_CUSTOMER_DEPOSITS"].replace(
        0,
        pd.NA
    )
)


# 5. NPL Ratio
df["NPL_RATIO"] = (
    df["EWAQ_NPL"]
    / df["EWAQ_GrossLoans"].replace(
        0,
        pd.NA
    )
)


# ============================================================
# HISTORICAL TREND FEATURES
# ============================================================

# Previous reporting-period values for each bank

df["PREV_LIQUIDITY_PROXY"] = (
    df.groupby("INSTITUTIONCODE")[
        "LIQUIDITY_PROXY"
    ].shift(1)
)

df["PREV_LOAN_TO_DEPOSIT"] = (
    df.groupby("INSTITUTIONCODE")[
        "LOAN_TO_DEPOSIT"
    ].shift(1)
)

df["PREV_NPL_RATIO"] = (
    df.groupby("INSTITUTIONCODE")[
        "NPL_RATIO"
    ].shift(1)
)

df["PREV_CASH_TO_LIQUID_LIAB"] = (
    df.groupby("INSTITUTIONCODE")[
        "CASH_TO_LIQUID_LIAB"
    ].shift(1)
)

df["PREV_LIQUID_ASSET_RATIO"] = (
    df.groupby("INSTITUTIONCODE")[
        "LIQUID_ASSET_TO_TOTAL_ASSET"
    ].shift(1)
)


# ============================================================
# CHANGE / TREND FEATURES
# ============================================================

df["LIQUIDITY_PROXY_CHANGE"] = (
    df["LIQUIDITY_PROXY"]
    - df["PREV_LIQUIDITY_PROXY"]
)

df["LOAN_TO_DEPOSIT_CHANGE"] = (
    df["LOAN_TO_DEPOSIT"]
    - df["PREV_LOAN_TO_DEPOSIT"]
)

df["NPL_RATIO_CHANGE"] = (
    df["NPL_RATIO"]
    - df["PREV_NPL_RATIO"]
)

df["CASH_LIQUIDITY_CHANGE"] = (
    df["CASH_TO_LIQUID_LIAB"]
    - df["PREV_CASH_TO_LIQUID_LIAB"]
)

df["LIQUID_ASSET_RATIO_CHANGE"] = (
    df["LIQUID_ASSET_TO_TOTAL_ASSET"]
    - df["PREV_LIQUID_ASSET_RATIO"]
)


# ============================================================
# HANDLE INFINITE / MISSING VALUES
# ============================================================

df = df.replace(
    [float("inf"), float("-inf")],
    pd.NA
)

df = df.dropna().reset_index(
    drop=True
)


# ============================================================
# SAVE CLEAN DATASET
# ============================================================

df.to_csv(
    CLEAN_FILE,
    index=False
)


# ============================================================
# CLEANING SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)

print(
    "Final shape:",
    df.shape
)

print(
    "Institutions:",
    df["INSTITUTIONCODE"].nunique()
)

print(
    "Date range:",
    df["REPORTINGDATE"].min(),
    "to",
    df["REPORTINGDATE"].max()
)

print("\nCurrent Target distribution:")

print(
    df["EWL_LIQUIDITY RATING"]
    .value_counts()
    .sort_index()
)

print("\nEngineered features:")

print([
    "LIQUIDITY_PROXY",
    "CASH_TO_LIQUID_LIAB",
    "LIQUID_ASSET_TO_TOTAL_ASSET",
    "LOAN_TO_DEPOSIT",
    "NPL_RATIO",
    "LIQUIDITY_PROXY_CHANGE",
    "LOAN_TO_DEPOSIT_CHANGE",
    "NPL_RATIO_CHANGE",
    "CASH_LIQUIDITY_CHANGE",
    "LIQUID_ASSET_RATIO_CHANGE"
])

print("\nClean dataset saved to:")
print(CLEAN_FILE)


# ============================================================
# NEXT-PERIOD TARGET
# ============================================================

df["TARGET_NEXT_RISK"] = (
    df.groupby("INSTITUTIONCODE")[
        "EWL_LIQUIDITY RATING"
    ].shift(-1)
)


# ============================================================
# REMOVE LAST RECORD OF EACH INSTITUTION
# ============================================================

df = df.dropna(
    subset=["TARGET_NEXT_RISK"]
).reset_index(drop=True)


df["TARGET_NEXT_RISK"] = (
    df["TARGET_NEXT_RISK"]
    .astype(int)
)


# ============================================================
# NEXT-PERIOD TARGET SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("NEXT-PERIOD TARGET")
print("=" * 70)

print(
    "\nNext-period target distribution:"
)

print(
    df["TARGET_NEXT_RISK"]
    .value_counts()
    .sort_index()
)


# ============================================================
# SAVE ML-READY DATASET
# ============================================================

df.to_csv(
    ML_OUTPUT,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ML DATASET READY")
print("=" * 70)

print(
    "ML-ready shape:",
    df.shape
)

print(
    "Institutions:",
    df["INSTITUTIONCODE"].nunique()
)

print(
    "Date range:",
    df["REPORTINGDATE"].min(),
    "to",
    df["REPORTINGDATE"].max()
)

print("\nML-ready dataset saved to:")
print(ML_OUTPUT)