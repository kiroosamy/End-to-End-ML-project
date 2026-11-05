import pandas as pd
import numpy as np

df = pd.read_csv("data/H2.csv", skipinitialspace=True)
df = df.apply(lambda c: c.str.strip() if pd.api.types.is_string_dtype(c) else c)
df = df.replace("NULL", np.nan)

print("Shape:", df.shape)
print()
print("Missing values:")
missing = df.isna().sum()
print(missing[missing > 0])
print()
print("Duplicates:", df.duplicated().sum())
print()
print("Target balance:")
print(df["IsCanceled"].value_counts(normalize=True))

print()
print("After dropping duplicates:")
df_clean = df.drop_duplicates()
print("Shape:", df_clean.shape)
print(df_clean["IsCanceled"].value_counts(normalize=True))

print()
dup_sizes = df.groupby(list(df.columns), dropna=False).size()
dup_sizes = dup_sizes[dup_sizes > 1]
print("Distinct rows that appear more than once:", len(dup_sizes))
print(dup_sizes.sort_values(ascending=False).head(5))