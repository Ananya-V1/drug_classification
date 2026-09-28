import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load results
X_pca = np.load("X_pca.npy")
drug_scaled = np.load("drug_scaled.npy")
drug_clusters = np.load("drug_clusters.npy")
results_df = pd.read_csv("results_task1.csv")
clf_df = pd.read_csv("results_task3.csv")
gi_final = pd.read_csv("gi_final.csv", index_col=0)

# Load tissue lookup from expression file
exp = pd.read_excel("RNA__Affy_HG_U133_Plus_2.0_RMA.xls", sheet_name="Results", skiprows=10, index_col=0)
cols = ["Gene name d", "Entrez gene id e", "Chromosome f", "Start f", "End f", "Cytoband f"]
tissue_lookup = {c.split(":")[1].strip(): c.split(":")[0] for c in exp.drop(columns=cols).columns}

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("NCI60 Drug Classification Pipeline", fontsize=16, fontweight="bold")

# Plot 1: PCA of cell lines colored by tissue type
ax = axes[0, 0]
tissues = [tissue_lookup.get(c, "Unknown") for c in gi_final.index]
palette = sns.color_palette("tab10", len(set(tissues)))
tissue_colors = {t: palette[i] for i, t in enumerate(sorted(set(tissues)))}
for i, (pc1, pc2) in enumerate(X_pca[:, :2]):
    ax.scatter(pc1, pc2, color=tissue_colors[tissues[i]], s=60)
handles = [plt.Line2D([0], [0], marker="o", color="w",
           markerfacecolor=tissue_colors[t], markersize=8, label=t)
           for t in sorted(set(tissues))]
ax.legend(handles=handles, fontsize=7)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title("Cell Lines by Tissue Type (PCA)")

# Plot 2: R² distribution
ax = axes[0, 1]
ax.hist(results_df["r2"], bins=20, color="steelblue", edgecolor="white")
ax.axvline(results_df["r2"].median(), color="red", linestyle="--",
           label=f"Median R²={results_df['r2'].median():.2f}")
ax.set_xlabel("R²"); ax.set_ylabel("Number of Drugs")
ax.set_title("Task 1: Drug Response Prediction (R²)")
ax.legend()

# Plot 3: Drug clusters in 2D
ax = axes[1, 0]
drug_pca2d = PCA(n_components=2, random_state=42).fit_transform(drug_scaled)
colors = ["steelblue", "darkorange"]
for k in range(2):
    mask = drug_clusters == k
    ax.scatter(drug_pca2d[mask, 0], drug_pca2d[mask, 1],
               color=colors[k], label=f"Cluster {k}", alpha=0.6, s=20)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title("Task 2: Drug Clusters")
ax.legend()

# Plot 4: AUC distribution
ax = axes[1, 1]
clf_df_clean = clf_df.dropna(subset=["auc"])
ax.hist(clf_df_clean["auc"], bins=20, color="darkorange", edgecolor="white")
ax.axvline(clf_df_clean["auc"].median(), color="red", linestyle="--",
           label=f"Median AUC={clf_df_clean['auc'].median():.2f}")
ax.axvline(0.5, color="gray", linestyle=":", label="Random (0.5)")
ax.set_xlabel("AUC"); ax.set_ylabel("Number of Drugs")
ax.set_title("Task 3: Drug Sensitivity Classification (AUC)")
ax.legend()

plt.tight_layout()
plt.savefig("results.png", dpi=150, bbox_inches="tight")
print("Saved: results.png")
plt.show()
