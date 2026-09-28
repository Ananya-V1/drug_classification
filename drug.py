import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

metadata = pd.read_excel("NCI60_CELL_LINE_METADATA.xls", sheet_name="clc", skiprows=10, index_col=0)
exp = pd.read_excel("RNA__Affy_HG_U133_Plus_2.0_RMA.xls", sheet_name="Results", skiprows=10, index_col=0)
gi = pd.read_csv("/Users/ananya/drug_classification/GI50.csv")

print(exp.shape)

# Change each cell line to be one row; long format to wide format
gi = gi.pivot_table(index="CELL_NAME", columns="NSC", values="AVERAGE")
# Drop drugs that have fewer than 50 values (tested on fewer than 50 cells)
gi = gi.dropna(thresh=50, axis=1)

# Removing these columns that are descriptions of genes and not the actual expression values
cols = ["Gene name d", "Entrez gene id e", "Chromosome f", "Start f", "End f", "Cytoband f"]
exp_clean = exp.drop(columns=cols)
# Remove the prefix part (before the :) so the names match
exp_clean.columns = [c.split(":")[1].strip() for c in exp_clean.columns]

# intersection between cell lines and expression
common = gi.index.intersection(exp_clean.columns)
print(f"Common cell lines: {len(common)}")

# New table with only intersection values
gi_final = gi.loc[common]
# X = the matching cell line columns flipped
# So rows = samples instead of = genes
X = exp_clean[common].T
# Reorder cell line rows to match GI50 order
X = X.reindex(gi_final.index)
# Convert values to float; if can't convert = NaN
X = X.apply(pd.to_numeric, errors="coerce")
# Replace NaN with median value of gene
# Dropping NaN columns would harm model
X = X.fillna(X.median())

print(all(X.index == gi_final.index))

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce to 50 components
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_.cumsum()
print(f"Variance explained by 50 PCs: {explained[-1]*100:.1f}%")

# Predict drug response
valid_drugs = []
for drug in gi_final.columns:
    y = gi_final[drug]
    # Less than 55 excluding NaN
    if y.notna().sum() < 55:
        continue
    # No variation in the data
    if y[y.notna()].std() < 0.2:
        continue
    valid_drugs.append(drug)

print(f"Valid drugs for modeling: {len(valid_drugs)}")

# Sample 100 drugs for speed
sample_drugs = pd.Series(valid_drugs).sample(100, random_state=42).tolist()

X_values = X.values  # raw gene expression matrix (59 x 54609)

results = []
for i, drug in enumerate(sample_drugs):
    y = gi_final[drug]
    # Convert real values to numpy
    y_clean = y[y.notna()].values
    X_drug = X_values[y.notna()]

    # Select top 100 genes most correlated with this drug's GI50 (fast)
    y_z = (y_clean - y_clean.mean()) / (y_clean.std() + 1e-8)
    X_z = (X_drug - X_drug.mean(axis=0)) / (X_drug.std(axis=0) + 1e-8)
    correlations = np.abs(X_z.T @ y_z) / len(y_z)
    top_genes = np.argsort(correlations)[-100:]
    X_selected = X_drug[:, top_genes]

    # Calculate r^2 score for each drug; ehich explains a percentage of the variation in the drug response
    scores = cross_val_score(RandomForestRegressor(n_estimators=50, random_state=42), X_selected, y_clean, cv=5, scoring="r2")
    results.append({"drug": drug, "r2": scores.mean()})
    print(f"  [{i+1}/100] drug {drug} — R²={scores.mean():.3f}")

results_df = pd.DataFrame(results).sort_values("r2", ascending=False)
print(results_df.head(10))
print(f"Drugs with R² > 0.3: {(results_df['r2'] > 0.3).sum()}")

# ── Task 2: Drug Clustering ──
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Each drug is a row, each column is a cell line's GI50 value
drug_matrix = gi_final[valid_drugs].dropna(axis=1).T
print(f"\nDrugs for clustering: {drug_matrix.shape[0]}")

# Scale
drug_scaled = StandardScaler().fit_transform(drug_matrix)

# Try k=2 to k=10 and find the best
for k in range(2, 11):
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(drug_scaled)
    score = silhouette_score(drug_scaled, labels)
    print(f"k={k} — silhouette={score:.3f}")

# Fit final model with best k
km_final = KMeans(n_clusters=2, random_state=42, n_init=10)
drug_clusters = km_final.fit_predict(drug_scaled)

# Add cluster labels to a dataframe
# Mapping each drug to cluster number
cluster_df = pd.DataFrame({
    "drug": drug_matrix.index,
    "cluster": drug_clusters
})

print(cluster_df["cluster"].value_counts())

# Look at average GI50 profile per cluster
drug_matrix["cluster"] = drug_clusters
print("\nMean GI50 per cluster:")
# Groups into two clusters: 0: weak drugs; 1: strong drugs
print(drug_matrix.groupby("cluster").mean().mean(axis=1))

# ── Task 3: Drug Sensitivity Classification ──
from sklearn.ensemble import RandomForestClassifier

clf_results = []
for i, drug in enumerate(sample_drugs):
    y = gi_final[drug]
    y_cont = y[y.notna()].values

    # Convert to binary: 1=sensitive, 0=resistant
    # Based on the median value of y_cont
    lb = LabelBinarizer()
    y_bin = lb.fit_transform((y_cont < np.median(y_cont)).astype(int)).ravel()

    # Select the top 100 genes that are correlated with the drug
    X_drug = X_values[y.notna()]
    y_z = (y_cont - y_cont.mean()) / (y_cont.std() + 1e-8)
    X_z = (X_drug - X_drug.mean(axis=0)) / (X_drug.std(axis=0) + 1e-8)
    correlations = np.abs(X_z.T @ y_z) / len(y_z)
    top_genes = np.argsort(correlations)[-100:]
    X_selected = X_drug[:, top_genes]

    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_selected, y_bin)

    scores = cross_val_score(rf, X_selected, y_bin, cv=5, scoring="roc_auc")
    clf_results.append({"drug": drug, "auc": scores.mean()})
    print(f"  [{i+1}/100] drug {drug} — AUC={scores.mean():.3f}")

clf_df = pd.DataFrame(clf_results).sort_values("auc", ascending=False)
print(clf_df.head(10))
print(f"Drugs with AUC > 0.7: {(clf_df['auc'] > 0.7).sum()}")

import shap

# Map probeset IDs to gene names using the annotation column
probeset_to_gene = exp["Gene name d"].to_dict()
# Use gene name if available, otherwise fall back to probeset ID
gene_names = [probeset_to_gene.get(pid, pid) for pid in X.columns.tolist()]

# SHAP for top 5 drugs by AUC
top5_drugs = clf_df.dropna(subset=["auc"]).head(5)["drug"].tolist()

for drug in top5_drugs:
    y = gi_final[drug]
    y_cont = y[y.notna()].values
    lb = LabelBinarizer()
    y_bin_shap = lb.fit_transform((y_cont < np.median(y_cont)).astype(int)).ravel()

    X_drug = X_values[y.notna()]
    y_z = (y_cont - y_cont.mean()) / (y_cont.std() + 1e-8)
    X_z = (X_drug - X_drug.mean(axis=0)) / (X_drug.std(axis=0) + 1e-8)
    correlations = np.abs(X_z.T @ y_z) / len(y_z)
    top_genes = np.argsort(correlations)[-100:]
    X_selected_shap = X_drug[:, top_genes]
    selected_gene_names = [gene_names[i] for i in top_genes]

    rf_shap = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_shap.fit(X_selected_shap, y_bin_shap)

    explainer = shap.TreeExplainer(rf_shap)
    shap_values = explainer.shap_values(X_selected_shap)
    sv = shap_values[:, :, 1]  # class 1 = sensitive
    shap.summary_plot(sv, X_selected_shap,
                      feature_names=selected_gene_names,
                      plot_type="dot",
                      title=f"SHAP — Drug {drug} (AUC={clf_df.loc[clf_df.drug==drug,'auc'].values[0]:.3f})",
                      show=False)
    plt.savefig(f"shap_drug_{drug}.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: shap_drug_{drug}.png")


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("NCI60 Drug Classification Pipeline", fontsize=16, fontweight="bold")

# Build tissue lookup from original expression column names (e.g. "BR:MCF7" -> "BR")
tissue_lookup = {c.split(":")[1].strip(): c.split(":")[0] for c in exp.drop(columns=cols).columns}

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

# Plot 2: R² distribution — Task 1
ax = axes[0, 1]
ax.hist(results_df["r2"], bins=20, color="steelblue", edgecolor="white")
ax.axvline(results_df["r2"].median(), color="red", linestyle="--",
           label=f"Median R²={results_df['r2'].median():.2f}")
ax.set_xlabel("R²"); ax.set_ylabel("Number of Drugs")
ax.set_title("Task 1: Drug Response Prediction (R²)")
ax.legend()

# Plot 3: Drug clusters in 2D — Task 2
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

# Plot 4: AUC distribution — Task 3
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
