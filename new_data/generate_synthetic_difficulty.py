
from pathlib import Path
import numpy as np
import pandas as pd

# Reproducible experiment
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

N_SAMPLES = 2000
N_FEATURES = 10
ANOMALY_RATE = 0.05

n_anomalies = int(N_SAMPLES * ANOMALY_RATE)
n_normal = N_SAMPLES - n_anomalies

# Generate the same underlying observations as the original script
normal_data = rng.normal(0.0, 1.0, (n_normal, N_FEATURES))
anomaly_base = rng.normal(0.0, 1.0, (n_anomalies, N_FEATURES))

normal_labels = np.zeros(n_normal, dtype=int)
anomaly_labels = np.ones(n_anomalies, dtype=int)

# Use the same shuffle order for both datasets
order = rng.permutation(N_SAMPLES)

output_dir = Path(__file__).resolve().parent.parent / "data"

for name, shift in [
    ("synthetic_moderate", 2.0),
    ("synthetic_hard", 1.0),
]:
    anomaly_data = anomaly_base.copy()
    anomaly_data[:, :3] += shift

    X = np.vstack([normal_data, anomaly_data])[order]
    y = np.concatenate([normal_labels, anomaly_labels])[order]

    df = pd.DataFrame(X)
    df[N_FEATURES] = y

    output_file = output_dir / f"{name}.csv"
    df.to_csv(output_file, index=False, header=False)

    print(f"\nDataset: {name}")
    print("Shape:", df.shape)
    print("Anomalies:", int(y.sum()))
    print("Shift:", shift)
    print("Missing values:", int(df.isna().sum().sum()))
    print("Saved to:", output_file)
