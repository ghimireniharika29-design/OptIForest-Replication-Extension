import pandas as pd
import matplotlib.pyplot as plt

# Load constructed dataset
df = pd.read_csv(
    "new_data/synthetic_anomaly.csv",
    header=None
)

# Assign column names
df.columns = [
    f"feature_{i+1}" for i in range(10)
] + ["label"]

normal = df[df["label"] == 0]
anomaly = df[df["label"] == 1]

# Plot two of the deliberately shifted features
plt.figure(figsize=(8, 6))

plt.scatter(
    normal["feature_1"],
    normal["feature_2"],
    alpha=0.35,
    s=18,
    label="Normal"
)

plt.scatter(
    anomaly["feature_1"],
    anomaly["feature_2"],
    alpha=0.8,
    s=30,
    label="Anomaly"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Constructed Anomaly Dataset")
plt.legend()
plt.tight_layout()

plt.savefig(
    "new_data/synthetic_preview.png",
    dpi=200
)

plt.show()