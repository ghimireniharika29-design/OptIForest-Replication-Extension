from pathlib import Path
import numpy as np
import pandas as pd

# Reproducibility
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# Dataset design
N_SAMPLES = 2000
N_FEATURES = 10
ANOMALY_RATE = 0.05

n_anomalies = int(N_SAMPLES * ANOMALY_RATE)
n_normal = N_SAMPLES - n_anomalies

# Generate normal observations
normal_data = rng.normal(
    loc=0.0,
    scale=1.0,
    size=(n_normal, N_FEATURES)
)

# Generate anomalous observations
anomaly_data = rng.normal(
    loc=0.0,
    scale=1.0,
    size=(n_anomalies, N_FEATURES)
)

# Shift the first three features for anomalous observations
anomaly_data[:, :3] += 4.0

# Ground-truth labels
# 0 = normal, 1 = anomaly
normal_labels = np.zeros(n_normal, dtype=int)
anomaly_labels = np.ones(n_anomalies, dtype=int)

# Combine data and labels
X = np.vstack([normal_data, anomaly_data])
y = np.concatenate([normal_labels, anomaly_labels])

# Shuffle the observations
order = rng.permutation(N_SAMPLES)
X = X[order]
y = y[order]

# Create dataframe
feature_names = [f"feature_{i+1}" for i in range(N_FEATURES)]

df = pd.DataFrame(X, columns=feature_names)
df["label"] = y

# Verify the constructed dataset
print("Dataset shape:", df.shape)

print("\nLabel counts:")
print(df["label"].value_counts().sort_index())

print("\nAnomaly percentage:")
print(df["label"].mean() * 100)

print("\nMissing values:")
print(df.isna().sum().sum())

print("\nMean of shifted features by class:")
print(
    df.groupby("label")[
        ["feature_1", "feature_2", "feature_3"]
    ].mean()
)

# Save in the format expected by OptIForest
output_file = Path(__file__).resolve().parent / "synthetic_anomaly.csv"

df.to_csv(
    output_file,
    index=False,
    header=False
)

print("\nSaved to:", output_file)