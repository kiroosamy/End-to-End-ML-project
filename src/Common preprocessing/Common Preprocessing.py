import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}

ADR_MAX_VALID = 1000 # Anything above this ADR is treated as a data-entry error (99.9th pct is ~290)


def to_snake(name: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return s.lower()


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def standardize_text(df: pd.DataFrame) -> pd.DataFrame:
    #Strip whitespace in text columns and turn 'NULL' strings into real NaN
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()
    df = df.replace({"NULL": pd.NA, "": pd.NA})
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    #Parse dates and build a proper ArrivalDate column
    df = df.copy()
    df["ReservationStatusDate"] = pd.to_datetime(df["ReservationStatusDate"])
    month_num = df["ArrivalDateMonth"].map(MONTHS)
    df["ArrivalDate"] = pd.to_datetime(dict(
        year=df["ArrivalDateYear"], month=month_num,
        day=df["ArrivalDateDayOfMonth"]))
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame: #here we will fill missing values using what the missingness actually means
    df = df.copy()
    #company: ~95% missing soo keeps only whether a company was involved
    df["HasCompany"] = df["Company"].notna().astype(int)
    df = df.drop(columns=["Company"])
    #agent: missing means booked without an agent
    df["Agent"] = df["Agent"].fillna("No Agent").astype(str)
    #country: tiny number missing
    df["Country"] = df["Country"].fillna("Unknown").astype(str)
    #children: a handful missing, assume none
    df["Children"] = df["Children"].fillna(0).astype(int)
    return df


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame: # here we will remove the rows that doesn't seems real
    guests = df["Adults"] + df["Children"] + df["Babies"]
    nights = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
    bad = (
        (guests == 0)
        | (nights == 0)
        | (df["ADR"] < 0)
        | (df["ADR"] > ADR_MAX_VALID)
    )
    return df.loc[~bad].copy()


def clean(df_raw: pd.DataFrame, drop_duplicates: bool = True,
          verbose: bool = True) -> pd.DataFrame:
    log = lambda msg, d: print(f"{msg:<38}{len(d):>8} rows") if verbose else None

    df = df_raw
    log("raw", df)
    df = standardize_text(df)
    df = fix_dtypes(df)
    df = handle_missing(df)
    df = remove_invalid_rows(df)
    log("after invalid-row removal", df)
    if drop_duplicates: #this prevents when we split the data into train/test that identical rows appear in both
        df = df.drop_duplicates().reset_index(drop=True)
        log("after duplicate removal", df)
    return df

def rename_snake_case(df: pd.DataFrame, mapping_path: str | None = None):
    #Optional: snake_case column names + a one-to-one mapping to source labels
    mapping = {c: to_snake(c) for c in df.columns}
    assert len(set(mapping.values())) == len(mapping), "name collision"
    if mapping_path:
        Path(mapping_path).write_text(json.dumps(mapping, indent=2))
    return df.rename(columns=mapping), mapping


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=r"C:\Users\PC\Downloads\H2.csv")
    ap.add_argument("--output", default="data/h2_clean.csv")
    ap.add_argument("--keep-duplicates", action="store_true")
    ap.add_argument("--snake-case", action="store_true")
    args = ap.parse_args()
    cleaned = clean(load_raw(args.input), drop_duplicates=not args.keep_duplicates)
    if args.snake_case:
        cleaned, _ = rename_snake_case(cleaned, "data/column_mapping.json")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output, index=False)
    print(f"saved -> {args.output}  shape={cleaned.shape}")