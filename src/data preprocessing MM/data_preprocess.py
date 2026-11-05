import pandas as pd

df = pd.read_csv("H2.csv")

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace({
        "NULL": pd.NA,
        "": pd.NA
    })

#removing unnecessary columns
df = df.drop(columns=[
    "ReservationStatus",
    "ReservationStatusDate",
    "AssignedRoomType",
    "Company"
])

#replace missing values
df["Children"] = df["Children"].fillna(0)

df["Country"] = df["Country"].fillna("Unknown")

df["Agent"] = df["Agent"].fillna("Unknown")

#ُencoding
categorical_cols = df.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

df = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)


df.to_csv("H2_preprocessed.csv", index=False)

print("Dataset shape:", df.shape)
