"""Very simple preprocessing + baseline model for H2 (predict IsCanceled)."""
import io
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "H2_csv.csv"
TARGET = "IsCanceled"

# Columns dropped, and why
DROP_COLS = [
    "ReservationStatus",       # leakage: it literally says Canceled / Check-Out
    "ReservationStatusDate",   # leakage: date the final status was set
    "AssignedRoomType",        # leakage-ish: only known at check-in
    "BookingChanges",          # leakage-ish: keeps changing until check-in
    "Agent",                   # ID column with many values
    "Company",                 # ID column, mostly missing
]


def load_data(path: str) -> pd.DataFrame:
    """Read the file, remove the quotes around each row, trim spaces, NULL -> NaN."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().replace('"', "")
    df = pd.read_csv(io.StringIO(text))
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].astype(str).str.strip()
    df = df.replace({"NULL": pd.NA, "nan": pd.NA, "": pd.NA})
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Simple cleaning that does not learn anything from the data."""
    df = df.copy()
    df = df.drop(columns=DROP_COLS)
    df = df.drop_duplicates()
    df["Children"] = pd.to_numeric(df["Children"]).fillna(0)
    df["Country"] = df["Country"].fillna("Unknown")
    df["TotalNights"] = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=100), cat_cols),
    ])


def main():
    df = clean_data(load_data(CSV_PATH))
    print("Shape after cleaning:", df.shape)
    print("Missing values left:", int(df.isna().sum().sum()))

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # Split first, so the scaler/encoder only learn from the training data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    model = Pipeline([
        ("prep", build_preprocessor(X_train)),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print("Baseline (always predict majority):", round(1 - y_test.mean(), 3))
    print("Accuracy:", round(accuracy_score(y_test, pred), 3))
    print("F1      :", round(f1_score(y_test, pred), 3))


if __name__ == "__main__":
    main()
