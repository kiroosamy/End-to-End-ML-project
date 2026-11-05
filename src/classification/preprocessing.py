import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_COLS = [
    "LeadTime", "ArrivalDateYear", "ArrivalDateDayOfMonth",
    "StaysInWeekendNights", "StaysInWeekNights",
    "Adults", "Children", "Babies",
    "IsRepeatedGuest", "PreviousCancellations",
    "PreviousBookingsNotCanceled", "DaysInWaitingList",
    "has_company",
]

CATEGORICAL_COLS = [
    "ArrivalDateMonth", "Meal", "Country", "MarketSegment",
    "DistributionChannel", "ReservedRoomType", "DepositType",
    "Agent", "CustomerType",
]


def clean_data(df):
    """Fixed rules only. Nothing here is learned from the data."""
    df = df.copy()
    df = df.apply(lambda c: c.str.strip() if pd.api.types.is_string_dtype(c) else c)
    df = df.replace("NULL", np.nan)

    df["has_company"] = df["Company"].notna().astype(int)
    df = df.drop(columns=["Company"])

    df["Agent"] = df["Agent"].apply(
        lambda v: "no_agent" if pd.isna(v) else str(int(float(v)))
    )
    return df


def build_preprocessor():
    """Learns its values from training data only, when fit is called."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=0.01,
        )),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_COLS),
        ("cat", categorical_pipe, CATEGORICAL_COLS),
    ])