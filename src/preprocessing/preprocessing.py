from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import joblib


# ---------------------------------------------------------------------
# 0. CONFIGURATION
# ---------------------------------------------------------------------

RANDOM_STATE = 42
EXPECTED_ROWS = 79_330
EXPECTED_COLUMNS = 31

# This file lives directly inside the project folder (H2_project/).
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "H2.csv"
OUTPUT_DIR = ROOT / "outputs" / "preprocessing"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
ARTIFACTS_DIR = ROOT / "artifacts" / "preprocessing"

for folder in [OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR, ARTIFACTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

sns.set_theme()


# ---------------------------------------------------------------------
# 1. LOAD DATA + VALIDATE
# ---------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Put H2.csv inside the project's data/ folder."
        )

    df = pd.read_csv(path)
    print("\n--- DATASET CHECK ---")
    print("Shape:", df.shape)
    print("First rows:\n", df.head())
    print("Info:")
    df.info()

    if df.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
        warnings.warn(
            f"Expected {(EXPECTED_ROWS, EXPECTED_COLUMNS)}, "
            f"but loaded {df.shape}. Verify that this is H2.csv."
        )

    required_columns = {
        "ADR", "IsCanceled", "LeadTime", "ReservationStatus",
        "ReservationStatusDate", "Agent", "Company", "Country"
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Required source columns are missing: {missing_columns}")

    return df


# ---------------------------------------------------------------------
# 2. DATA QUALITY REPORT
# ---------------------------------------------------------------------

def create_data_quality_report(df: pd.DataFrame) -> None:
    duplicate_count = int(df.duplicated().sum())

    # Measure the size of the duplicate problem (on the raw data).
    dup_mask = df.duplicated(keep=False)
    print("Duplicate rows (all copies):", int(dup_mask.sum()))
    print("Extra copies:", duplicate_count)
    print("Cancel rate in duplicates:", df.loc[dup_mask, "IsCanceled"].mean())
    print("Cancel rate overall:", df["IsCanceled"].mean())

    missing_report = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percentage": df.isna().mean().mul(100).round(2),
        "unique_values": df.nunique(dropna=True),
        "dtype": df.dtypes.astype(str)
    }).sort_values("missing_percentage", ascending=False)

    missing_report.to_csv(REPORTS_DIR / "missing_values_report.csv")

    (REPORTS_DIR / "duplicates_report.txt").write_text(
        f"Exact duplicate rows (extra copies): {duplicate_count}\n"
        f"Rows involved in duplication (all copies): {int(dup_mask.sum())}\n"
        "Decision: exact duplicates are dropped in clean_data() before the "
        "train/test split to avoid train/test leakage.\n",
        encoding="utf-8"
    )

    df.describe(include="all").T.to_csv(
        REPORTS_DIR / "summary_statistics.csv"
    )

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    cardinality = pd.DataFrame({
        "unique_categories": df[categorical_columns].nunique(dropna=True),
        "missing_count": df[categorical_columns].isna().sum()
    }).sort_values("unique_categories", ascending=False)

    cardinality.to_csv(REPORTS_DIR / "categorical_cardinality.csv")

    df.dtypes.astype(str).rename("dtype").to_csv(
        REPORTS_DIR / "data_types.csv"
    )

    print("\n--- DATA QUALITY ---")
    print(missing_report)
    print("Exact duplicate rows:", duplicate_count)


# ---------------------------------------------------------------------
# 3. CHECK UNUSUAL VALUES
# ---------------------------------------------------------------------

def check_unusual_values(df: pd.DataFrame) -> pd.DataFrame:
    checks = []

    # Nonnegative count/amount-like columns
    nonnegative_columns = [
        "LeadTime", "StaysInWeekendNights", "StaysInWeekNights",
        "Adults", "Children", "Babies", "PreviousCancellations",
        "PreviousBookingsNotCanceled", "BookingChanges",
        "DaysInWaitingList", "RequiredCarParkingSpaces",
        "TotalOfSpecialRequests"
    ]

    for col in nonnegative_columns:
        if col in df.columns:
            count = int((df[col].dropna() < 0).sum())
            checks.append({
                "column": col,
                "check": "negative values",
                "count": count
            })

    if "ADR" in df.columns:
        checks.append({
            "column": "ADR",
            "check": "zero or negative ADR",
            "count": int((df["ADR"].dropna() <= 0).sum())
        })

    if "IsCanceled" in df.columns:
        checks.append({
            "column": "IsCanceled",
            "check": "values outside {0, 1}",
            "count": int((~df["IsCanceled"].dropna().isin([0, 1])).sum())
        })

    if "IsRepeatedGuest" in df.columns:
        checks.append({
            "column": "IsRepeatedGuest",
            "check": "values outside {0, 1}",
            "count": int((~df["IsRepeatedGuest"].dropna().isin([0, 1])).sum())
        })

    result = pd.DataFrame(checks)
    result.to_csv(REPORTS_DIR / "unusual_values_report.csv", index=False)
    print("\n--- UNUSUAL VALUES ---")
    print(result)
    return result


# ---------------------------------------------------------------------
# 4. OUTLIER REPORT (IQR FLAGGING; DOES NOT DELETE OUTLIERS)
# ---------------------------------------------------------------------

def create_outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=np.number)
    rows = []

    for col in numeric.columns:
        values = numeric[col].dropna()
        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flagged = ((values < lower) | (values > upper)).sum()

        rows.append({
            "column": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_iqr_bound": lower,
            "upper_iqr_bound": upper,
            "outlier_count": int(flagged),
            "outlier_percentage": round(100 * flagged / len(values), 2)
        })

    report = pd.DataFrame(rows).sort_values(
        "outlier_percentage", ascending=False
    )
    report.to_csv(REPORTS_DIR / "outlier_report_iqr.csv", index=False)
    return report


# ---------------------------------------------------------------------
# 5. EDA PLOTS
# ---------------------------------------------------------------------

def create_eda_plots(df: pd.DataFrame) -> None:
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    # Histograms
    for col in numeric_columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), bins=40, kde=False)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"hist_{col}.png", dpi=150)
        plt.close()

    # Boxplots
    for col in numeric_columns:
        plt.figure(figsize=(8, 3))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"box_{col}.png", dpi=150)
        plt.close()

    # Correlation heatmap
    if len(numeric_columns) >= 2:
        corr = df[numeric_columns].corr()
        corr.to_csv(REPORTS_DIR / "numeric_correlation.csv")

        plt.figure(figsize=(14, 10))
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Numeric Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=150)
        plt.close()

    # Top categorical frequencies
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in categorical_columns:
        top = df[col].fillna("Missing").value_counts().head(15)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=top.values, y=top.index)
        plt.title(f"Top categories: {col}")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"category_{col}.png", dpi=150)
        plt.close()


# ---------------------------------------------------------------------
# 6. CONSERVATIVE CLEANING
# ---------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    # Normalize whitespace in string columns, without changing source labels.
    object_columns = cleaned.select_dtypes(include="object").columns
    for col in object_columns:
        cleaned[col] = cleaned[col].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    # Parse date with invalid values converted to NaT.
    if "ReservationStatusDate" in cleaned.columns:
        cleaned["ReservationStatusDate"] = pd.to_datetime(
            cleaned["ReservationStatusDate"], errors="coerce"
        )

    # Outliers are not dropped; exact duplicates are dropped before splitting.
    # Target values are never imputed.
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    print(f"Dropped {before - len(cleaned)} duplicate rows")

    cleaned.to_csv(OUTPUT_DIR / "h2_cleaned_conservative.csv", index=False)
    return cleaned


# ---------------------------------------------------------------------
# 7. FEATURE ENGINEERING
# ---------------------------------------------------------------------

class BookingFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    DataFrame transformer. Creates total nights and total guests.
    Expects source columns to be present.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if {
            "StaysInWeekendNights", "StaysInWeekNights"
        }.issubset(X.columns):
            X["TotalNights"] = (
                X["StaysInWeekendNights"].fillna(0)
                + X["StaysInWeekNights"].fillna(0)
            )

        if {"Adults", "Children", "Babies"}.issubset(X.columns):
            X["TotalGuests"] = (
                X["Adults"].fillna(0)
                + X["Children"].fillna(0)
                + X["Babies"].fillna(0)
            )

        if "ArrivalDateMonth" in X.columns:
            month_map = {
                "January": 1, "February": 2, "March": 3,
                "April": 4, "May": 5, "June": 6,
                "July": 7, "August": 8, "September": 9,
                "October": 10, "November": 11, "December": 12
            }
            X["ArrivalMonthNumber"] = X["ArrivalDateMonth"].map(month_map)

        return X


# ---------------------------------------------------------------------
# 8. TASK-SPECIFIC FEATURE LISTS + LEAKAGE AUDIT
# ---------------------------------------------------------------------

REGRESSION_TARGET = "ADR"
CLASSIFICATION_TARGET = "IsCanceled"

# Candidate features available at booking creation; review with instructor.
REGRESSION_FEATURES = [
    "LeadTime", "ArrivalDateYear", "ArrivalDateMonth",
    "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth",
    "StaysInWeekendNights", "StaysInWeekNights",
    "Adults", "Children", "Babies", "Meal", "Country",
    "MarketSegment", "DistributionChannel", "IsRepeatedGuest",
    "PreviousCancellations", "PreviousBookingsNotCanceled",
    "ReservedRoomType", "DepositType", "CustomerType",
    "RequiredCarParkingSpaces", "TotalOfSpecialRequests"
]

CLASSIFICATION_FEATURES = [
    "LeadTime", "ArrivalDateYear", "ArrivalDateMonth",
    "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth",
    "StaysInWeekendNights", "StaysInWeekNights",
    "Adults", "Children", "Babies", "Meal", "Country",
    "MarketSegment", "DistributionChannel", "IsRepeatedGuest",
    "PreviousCancellations", "PreviousBookingsNotCanceled",
    "ReservedRoomType", "DepositType", "CustomerType",
    "RequiredCarParkingSpaces", "TotalOfSpecialRequests"
]

# Clustering uses booking attributes, not final outcomes.
CLUSTERING_FEATURES = [
    "LeadTime", "StaysInWeekendNights", "StaysInWeekNights",
    "Adults", "Children", "Babies", "ADR",
    "RequiredCarParkingSpaces", "TotalOfSpecialRequests"
]

LEAKAGE_AUDIT = {
    "ReservationStatus": (
        "Exclude from IsCanceled prediction: it directly reveals the final "
        "booking outcome."
    ),
    "ReservationStatusDate": (
        "Exclude from early prediction: it is tied to the final status date."
    ),
    "IsCanceled": (
        "Target for classification; exclude from predictors and clustering."
    ),
    "BookingChanges": (
        "May occur after booking creation; include only if prediction point "
        "is after changes have been observed."
    ),
    "DaysInWaitingList": (
        "May not be known at booking creation; use only if available at the "
        "defined prediction point."
    ),
    "AssignedRoomType": (
        "May be assigned after booking; assess availability at prediction time."
    ),
    "Agent_and_Company": (
        "Identifiers; do not treat as continuous numeric quantities. "
        "High cardinality and missingness require a justified strategy."
    ),
    "ADR_in_clustering": (
        "Permissible only if clustering is intended to include booking price. "
        "Exclude if price should not influence discovered groups."
    )
}


def validate_feature_lists(df: pd.DataFrame) -> dict:
    feature_sets = {
        "regression": REGRESSION_FEATURES,
        "classification": CLASSIFICATION_FEATURES,
        "clustering": CLUSTERING_FEATURES
    }

    checked = {}
    for task, columns in feature_sets.items():
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"{task}: source columns missing: {missing}")
        checked[task] = columns.copy()

    (REPORTS_DIR / "leakage_audit.json").write_text(
        json.dumps(LEAKAGE_AUDIT, indent=4),
        encoding="utf-8"
    )

    (REPORTS_DIR / "feature_selection.json").write_text(
        json.dumps(checked, indent=4),
        encoding="utf-8"
    )

    return checked


# ---------------------------------------------------------------------
# 9. REUSABLE PREPROCESSING PIPELINE
# ---------------------------------------------------------------------

def build_preprocessor(X: pd.DataFrame, scale_numeric=True) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        ))
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features)
        ],
        remainder="drop"
    )


def build_task_pipeline(df: pd.DataFrame, feature_columns: list,
                        target_column: str | None,
                        scale_numeric=True) -> Pipeline:
    X = df[feature_columns].copy()

    if target_column is not None:
        # target is kept separate; no target imputation
        pass

    return Pipeline(steps=[
        ("feature_engineering", BookingFeatureEngineer()),
        ("preprocessor", build_preprocessor(
            BookingFeatureEngineer().fit_transform(X),
            scale_numeric=scale_numeric
        ))
    ])


def save_preprocessing_pipelines(df: pd.DataFrame, feature_sets: dict) -> None:
    """
    Build and save unfitted pipeline templates.
    These are templates only; no training/model fitting is performed.
    Fitting on training data will happen in the modeling milestone.
    """
    tasks = {
        "regression": {
            "features": feature_sets["regression"],
            "target": REGRESSION_TARGET,
            "scale": True
        },
        "classification": {
            "features": feature_sets["classification"],
            "target": CLASSIFICATION_TARGET,
            "scale": True
        },
        "clustering": {
            "features": feature_sets["clustering"],
            "target": None,
            "scale": True
        }
    }

    for task, config in tasks.items():
        X = df[config["features"]].copy()
        engineered_X = BookingFeatureEngineer().fit_transform(X)
        pipeline = Pipeline(steps=[
            ("feature_engineering", BookingFeatureEngineer()),
            ("preprocessor", build_preprocessor(
                engineered_X,
                scale_numeric=config["scale"]
            ))
        ])

        joblib.dump(
            pipeline,
            ARTIFACTS_DIR / f"{task}_preprocessing_template.joblib"
        )

        # Save a simple schema/config note alongside each artifact.
        schema = {
            "task": task,
            "target": config["target"],
            "input_features": config["features"],
            "scaling": config["scale"],
            "status": "unfitted template; fit only on training data later"
        }
        (ARTIFACTS_DIR / f"{task}_schema.json").write_text(
            json.dumps(schema, indent=4),
            encoding="utf-8"
        )


# ---------------------------------------------------------------------
# 10. PREPROCESSING SUMMARY
# ---------------------------------------------------------------------

def write_summary(df: pd.DataFrame) -> None:
    summary = f"""
CSE382 Milestone 1 - Preprocessing Summary
==========================================

Dataset: H2.csv
Rows (after duplicate removal): {df.shape[0]}
Columns: {df.shape[1]}

Tasks prepared:
- Regression target: ADR
- Classification target: IsCanceled
- Clustering: unsupervised; no target

Important:
- No models were trained.
- No target values were imputed.
- Outliers were reported, not deleted. Exact duplicates were dropped before splitting.
- Preprocessing templates are saved unfitted.
- In the modeling stage, split the data first and fit imputation/encoding/scaling
  on training data only.
- Review the prediction point and leakage audit with the instructor/team.
"""
    (REPORTS_DIR / "preprocessing_summary.txt").write_text(
        summary.strip() + "\n",
        encoding="utf-8"
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    df = load_data(DATA_PATH)

    create_data_quality_report(df)
    check_unusual_values(df)
    create_outlier_report(df)
    create_eda_plots(df)

    cleaned_df = clean_data(df)

    feature_sets = validate_feature_lists(cleaned_df)
    save_preprocessing_pipelines(cleaned_df, feature_sets)
    write_summary(cleaned_df)

    print("\nFinished preprocessing stage.")
    print("Reports:", REPORTS_DIR)
    print("Figures:", FIGURES_DIR)
    print("Artifacts:", ARTIFACTS_DIR)
    print("No ML models were trained.")


if __name__ == "__main__":
    main()