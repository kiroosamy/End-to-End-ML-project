import numpy as np
import pandas as pd

from preprocessing import clean_data, build_preprocessor

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

X_train = clean_data(train.drop(columns=["IsCanceled"]))
X_test = clean_data(test.drop(columns=["IsCanceled"]))

prep = build_preprocessor()
Xt_train = prep.fit_transform(X_train)
Xt_test = prep.transform(X_test)

print("Train before:", X_train.shape)
print("Train after:", Xt_train.shape)
print("Test after:", Xt_test.shape)

dense = Xt_train.toarray() if hasattr(Xt_train, "toarray") else Xt_train
print("Missing values left in train:", int(np.isnan(dense).sum()))