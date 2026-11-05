import pandas as pd
import numpy as np

df = pd.read_csv("data/H2.csv", skipinitialspace=True)
df = df.apply(lambda c: c.str.strip() if pd.api.types.is_string_dtype(c) else c)
df = df.replace("NULL", np.nan)
df = df.drop_duplicates()

print("ReservationStatus vs IsCanceled:")
print(pd.crosstab(df["ReservationStatus"], df["IsCanceled"]))

print()
print("Cancel rate by BookingChanges:")
print(df.groupby("BookingChanges")["IsCanceled"].mean().head(6))

print()
print("Cancel rate by DepositType:")
print(df.groupby("DepositType")["IsCanceled"].mean())