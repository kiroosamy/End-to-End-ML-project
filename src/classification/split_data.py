import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from features import TARGET, DROP_COLUMNS

SEED = 42

df = pd.read_csv("data/H2.csv", skipinitialspace=True)
df = df.apply(lambda c: c.str.strip() if pd.api.types.is_string_dtype(c) else c)
df = df.replace("NULL", np.nan)
df = df.drop_duplicates()

X = df.drop(columns=[TARGET] + DROP_COLUMNS)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)
print()
print("Cancel rate in train:", round(y_train.mean(), 4))
print("Cancel rate in test:", round(y_test.mean(), 4))
print()
print("Columns used:")
print(list(X_train.columns))

X_train.assign(IsCanceled=y_train).to_csv("data/train.csv", index=False)
X_test.assign(IsCanceled=y_test).to_csv("data/test.csv", index=False)
