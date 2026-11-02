
import matplotlib.pyplot as plt
import numpy as np

# Mean results from the 15 experimental runs
datasets = ["Easy (+4)", "Moderate (+2)", "Hard (+1)"]
roc_auc = [0.9999975439, 0.9588, 0.7102]
average_precision = [0.9999535300, 0.7992, 0.1937]

x = np.arange(len(datasets))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 4.5))

bars_roc = ax.bar(
    x - width / 2, roc_auc, width,
    label="ROC AUC", color="#3268A8"
)
bars_ap = ax.bar(
    x + width / 2, average_precision, width,
    label="Average Precision", color="#E49A45"
)

ax.set_ylabel("Mean score")
ax.set_xlabel("Synthetic anomaly difficulty")
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylim(0, 1.12)
ax.legend(frameon=False)

for bars in [bars_roc, bars_ap]:
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(
    "new_data/synthetic_results_comparison.pdf",
    bbox_inches="tight"
)
plt.savefig(
    "new_data/synthetic_results_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

print("Synthetic results comparison figure saved.")
