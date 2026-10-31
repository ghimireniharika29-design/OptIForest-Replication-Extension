from pathlib import Path

import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import OneHotEncoder


# -----------------------------
# File locations
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "ADBenchmarks-anomaly-detection-datasets"
    / "categorical data"
    / "cmc-nominal.arff"
)

OUTPUT_FILE = BASE_DIR / "cmc_processed.csv"


# -----------------------------
# 1. Load original ARFF dataset
# -----------------------------
data, meta = arff.loadarff(INPUT_FILE)
df = pd.DataFrame(data)

# Decode byte strings produced by scipy.io.arff
for column in df.columns:
    if df[column].dtype == object:
        df[column] = df[column].str.decode("utf-8")


# -----------------------------
# 2. Separate predictors / label
# -----------------------------
label_column = "class_numberofchildren"

X = df.drop(columns=[label_column])
y = df[label_column].astype(int)


# -----------------------------
# 3. One-hot encode categorical predictors
# -----------------------------
encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

X_encoded = encoder.fit_transform(X)

encoded_columns = encoder.get_feature_names_out(X.columns)

X_encoded_df = pd.DataFrame(
    X_encoded,
    columns=encoded_columns
)


# -----------------------------
# 4. Put anomaly label last
# -----------------------------
processed = X_encoded_df.copy()
processed["label"] = y.to_numpy()


# -----------------------------
# 5. Verify processed data
# -----------------------------
print("Original shape:", df.shape)
print("Processed shape:", processed.shape)

print("\nLabel counts:")
print(processed["label"].value_counts().sort_index())

print("\nMissing values:", processed.isna().sum().sum())

print("\nEncoded feature names:")
print(encoded_columns.tolist())


# -----------------------------
# 6. Save without header/index
# OptIForest demo.py uses header=None
# -----------------------------
processed.to_csv(
    OUTPUT_FILE,
    index=False,
    header=False
)

print(f"\nSaved to: {OUTPUT_FILE}")