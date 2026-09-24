import pandas as pd
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIG
# ============================================================

DATA_FILE = Path("data/ml_ready_bank_liquidity.csv")

TRAIN_END_DATE = "2019-12-31"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

df["REPORTINGDATE"] = pd.to_datetime(
    df["REPORTINGDATE"]
)

print("Dataset shape:", df.shape)


# ============================================================
# CREATE PROJECT-DEFINED RISK BANDS
# ============================================================

def create_risk_band(rating):
    if rating <= 1:
        return 0       # Low
    elif rating <= 3:
        return 1       # Moderate
    else:
        return 2       # High


df["TARGET_RISK_BAND"] = (
    df["TARGET_NEXT_RISK"]
    .apply(create_risk_band)
)


# ============================================================
# SORT DATA
# ============================================================

df = df.sort_values(
    ["REPORTINGDATE", "INSTITUTIONCODE"]
).reset_index(drop=True)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [

    # Liquidity
    "13_CASH",
    "22_TREASURY_BILLS",
    "23_OTHER_GOV_SECURITIES",
    "XX_CUSTOMER_DEPOSITS",
    "XX_TOTAL_LIQUID_LIAB",
    "XX_TOTAL_LIQUID_ASSET",
    "XX_MLA",

    # Balance Sheet
    "F077_ASSETS_TOTAL",
    "F125_LIAB_TOTAL",

    # Credit Risk
    "EWAQ_GrossLoans",
    "EWAQ_Capital",
    "EWAQ_NPL",
    "EWAQ_NPLsNetOfProvisions",
    "EWAQ_NPLsNetOfProvisions2CoreCapital",

    # Existing Indicators
    "LR",
    "DR",
    "IBCM",

    # Macro
    "GDP",
    "INF",

    # Liquidity Ratios
    "LIQUIDITY_PROXY",
    "CASH_TO_LIQUID_LIAB",
    "LIQUID_ASSET_TO_TOTAL_ASSET",
    "LOAN_TO_DEPOSIT",
    "NPL_RATIO",

    # Trend Features
    "LIQUIDITY_PROXY_CHANGE",
    "LOAN_TO_DEPOSIT_CHANGE",
    "NPL_RATIO_CHANGE",
    "CASH_LIQUIDITY_CHANGE",
    "LIQUID_ASSET_RATIO_CHANGE"
]


TARGET = "TARGET_RISK_BAND"


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

train_df = df[
    df["REPORTINGDATE"] <= TRAIN_END_DATE
].copy()

test_df = df[
    df["REPORTINGDATE"] > TRAIN_END_DATE
].copy()


print("\n" + "=" * 70)
print("TIME-BASED TRAIN / TEST SPLIT")
print("=" * 70)

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

print(
    "Training period:",
    train_df["REPORTINGDATE"].min(),
    "to",
    train_df["REPORTINGDATE"].max()
)

print(
    "Testing period:",
    test_df["REPORTINGDATE"].min(),
    "to",
    test_df["REPORTINGDATE"].max()
)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("RISK BAND DISTRIBUTION")
print("=" * 70)

print(
    df[TARGET]
    .value_counts()
    .sort_index()
)


# ============================================================
# CREATE X AND y
# ============================================================

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train
)

print("Training complete!")


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("3-CLASS RISK BAND MODEL PERFORMANCE")
print("=" * 70)

print(f"Accuracy          : {accuracy:.4f}")
print(f"Weighted Precision: {precision:.4f}")
print(f"Weighted Recall   : {recall:.4f}")
print(f"Weighted F1       : {weighted_f1:.4f}")
print(f"Macro F1          : {macro_f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Low Risk",
            "Moderate Risk",
            "High Risk"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 70)
print("TOP 15 RISK DRIVERS")
print("=" * 70)

print(
    importance
    .head(15)
    .to_string(index=False)
)


# ============================================================
# SAVE TRAINED MODEL
# ============================================================

MODEL_OUTPUT = Path("ml/liquidity_risk_model.joblib")

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "risk_bands": {
            0: "Low Risk",
            1: "Moderate Risk",
            2: "High Risk"
        }
    },
    MODEL_OUTPUT
)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)
print("Model saved to:")
print(MODEL_OUTPUT)