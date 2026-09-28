import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load prepped data
X_values = np.load("X_values.npy")
X_pca = np.load("X_pca.npy")
valid_drugs = np.load("valid_drugs.npy", allow_pickle=True).tolist()
sample_drugs = np.load("sample_drugs.npy", allow_pickle=True).tolist()
gene_names = np.load("gene_names.npy", allow_pickle=True).tolist()
gi_final = pd.read_csv("gi_final.csv", index_col=0)

print(f"Loaded: {X_values.shape[0]} cell lines, {len(valid_drugs)} valid drugs")

# Task 1: Drug Response Prediction (Regression)
print("\nTask 1: Drug Response Prediction...")
results = []
for i, drug in enumerate(sample_drugs):
    y = gi_final[str(drug)] # ignore rows with na values in y
    y_clean = y[y.notna()].values
    X_drug = X_values[y.notna()]

    # Select top 100 genes most correlated with this drug's GI50
    y_z = (y_clean - y_clean.mean()) / (y_clean.std() + 1e-8) # standardize y and x values into z scores
    X_z = (X_drug - X_drug.mean(axis=0)) / (X_drug.std(axis=0) + 1e-8)
    correlations = np.abs(X_z.T @ y_z) / len(y_z) # compute the absolute Pearson correlation (measures strength of linear relationship between two variables)
    top_genes = np.argsort(correlations)[-100:] # find top 100 correlations (argsort orders smallest to largest so find top 100 from back to front)
    X_selected = X_drug[:, top_genes] # filter to only the x of top genes

    # random forest regressor to predict the gi50 value (concentration of the drug) needed for each cell line
    # basically testing how the cell line's gene expression (independent of the drug) will respond when exposed to the drug
    scores = cross_val_score(RandomForestRegressor(n_estimators=50, random_state=42),
                             X_selected, y_clean, cv=5, scoring="r2")
    results.append({"drug": drug, "r2": scores.mean()})
    print(f"  [{i+1}/100] drug {drug} — R²={scores.mean():.3f}")

results_df = pd.DataFrame(results).sort_values("r2", ascending=False)
print(results_df.head(10)) # top ten drugs with r^2 higher than 0.3
print(f"Drugs with R² > 0.3: {(results_df['r2'] > 0.3).sum()}") # R^2 shows amount of variance the model captures/higher R^2 = better fit for the model
results_df.to_csv("results_task1.csv", index=False)

# Task 2: Drug Clustering
print("\nTask 2: Drug Clustering...")
valid_drugs_str = [str(drug) for drug in valid_drugs]
existing_columns = [col for col in valid_drugs_str if col in gi_final.columns]

print(f"Matching columns found in gi_final: {len(existing_columns)} out of {len(valid_drugs)}")

# If we found matching string columns, slice them. Otherwise, try row slicing (index).
if len(existing_columns) > 0:
    drug_matrix = gi_final[existing_columns].dropna(axis=1).T
else:
    # Fallback: Check if the drug IDs are actually row indices instead of columns
    existing_rows = [idx for idx in valid_drugs if idx in gi_final.index]
    if len(existing_rows) > 0:
        print(f"Matching rows found in gi_final index: {len(existing_rows)}")
        drug_matrix = gi_final.loc[existing_rows].dropna(axis=1).T
    else:
        # If both fail, print the structural details so we can diagnose the layout mismatch
        print("--- Error Diagnosis ---")
        print("gi_final columns (first 5):", list(gi_final.columns[:5]))
        print("gi_final index (first 5):", list(gi_final.index[:5]))
        raise KeyError("Could not find valid_drugs as either rows or columns in gi_final.")
drug_matrix = gi_final[existing_columns].dropna(axis=1).T
from sklearn.preprocessing import StandardScaler
drug_scaled = StandardScaler().fit_transform(drug_matrix)

# Find best k
for k in range(2, 11):
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(drug_scaled)
    score = silhouette_score(drug_scaled, labels) #silhoutte score to find optimal number of k clusters
    print(f"k={k} — silhouette={score:.3f}")

# Fit final model with best k=2
km_final = KMeans(n_clusters=2, random_state=42, n_init=10)
drug_clusters = km_final.fit_predict(drug_scaled)

cluster_df = pd.DataFrame({"drug": drug_matrix.index, "cluster": drug_clusters})
print(cluster_df["cluster"].value_counts())

drug_matrix["cluster"] = drug_clusters
print("\nMean GI50 per cluster:")
print(drug_matrix.groupby("cluster").mean().mean(axis=1))

np.save("drug_scaled.npy", drug_scaled)
np.save("drug_clusters.npy", drug_clusters)
cluster_df.to_csv("results_task2.csv", index=False)

# Task 3: Drug Sensitivity Classification
print("\nTask 3: Drug Sensitivity Classification...")
clf_results = []
for i, drug in enumerate(sample_drugs):
    y = gi_final[str(drug)]
    y_cont = y[y.notna()].values

    lb = LabelBinarizer()
    # binarize responses (y): 0 if less than median/resistant and 1 ir greater than or equal/sensitive
    y_bin = lb.fit_transform((y_cont < np.median(y_cont)).astype(int)).ravel()

    X_drug = X_values[y.notna()]
    # Select top 100 genes most correlated with this drug's GI50
    
    y_z = (y_cont - y_cont.mean()) / (y_cont.std() + 1e-8)
    X_z = (X_drug - X_drug.mean(axis=0)) / (X_drug.std(axis=0) + 1e-8)
    correlations = np.abs(X_z.T @ y_z) / len(y_z)
    top_genes = np.argsort(correlations)[-100:]
    X_selected = X_drug[:, top_genes]

    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_selected, y_bin)

    scores = cross_val_score(rf, X_selected, y_bin, cv=5, scoring="roc_auc")
    clf_results.append({"drug": drug, "auc": scores.mean()})
    print(f"  [{i+1}/100] drug {drug} — AUC={scores.mean():.3f}") # AUC for how well model can differentiate between two classes for each drug

clf_df = pd.DataFrame(clf_results).sort_values("auc", ascending=False)
print(clf_df.head(10))
print(f"Drugs with AUC > 0.7: {(clf_df['auc'] > 0.7).sum()}")
clf_df.to_csv("results_task3.csv", index=False)

# SHAP for top 5 drugs
print("\nSHAP analysis for top 5 drugs...")
top5_drugs = clf_df.dropna(subset=["auc"]).head(5)["drug"].tolist()

for drug in top5_drugs:
    y = gi_final[str(drug)]
    y_cont = y[y.notna()].values
    # binarize responses (y): 0 if less than median/resistant and 1 ir greater than or equal/sensitive
    lb = LabelBinarizer()
    y_bin_shap = lb.fit_transform((y_cont < np.median(y_cont)).astype(int)).ravel()

    X_drug = X_values[y.notna()]
    y_z = (y_cont - y_cont.mean()) / (y_cont.std() + 1e-8)
    X_z = (X_drug - X_drug.mean(axis=0)) / (X_drug.std(axis=0) + 1e-8)
    correlations = np.abs(X_z.T @ y_z) / len(y_z)
    top_genes = np.argsort(correlations)[-100:]
    X_selected_shap = X_drug[:, top_genes]
    selected_gene_names = [gene_names[i] for i in top_genes]

    # random forest classifier to classify whether cell lines are resistant or sensitive to the drug
    rf_shap = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_shap.fit(X_selected_shap, y_bin_shap)

    explainer = shap.TreeExplainer(rf_shap)
    shap_values = explainer.shap_values(X_selected_shap)
    sv = shap_values[:, :, 1]
    shap.summary_plot(sv, X_selected_shap,
                      feature_names=selected_gene_names,
                      plot_type="dot",
                      title=f"SHAP — Drug {drug} (AUC={clf_df.loc[clf_df.drug==drug,'auc'].values[0]:.3f})",
                      show=False)
    plt.savefig(f"shap_drug_{drug}.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: shap_drug_{drug}.png")

print("\nAll results saved.")
