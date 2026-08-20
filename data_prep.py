import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data
metadata = pd.read_excel("NCI60_CELL_LINE_METADATA.xls", sheet_name="clc", skiprows=10, index_col=0)
exp = pd.read_excel("RNA__Affy_HG_U133_Plus_2.0_RMA.xls", sheet_name="Results", skiprows=10, index_col=0)
gi = pd.read_csv("/Users/ananya/drug_classification/GI50.csv")

print(f"Expression matrix: {exp.shape}")


# Change each cell line to be one row; long format to wide format
gi = gi.pivot_table(index="CELL_NAME", columns="NSC", values="AVERAGE")
# Drop drugs that have fewer than 50 values (tested on fewer than 50 cells)
gi = gi.dropna(thresh=50, axis=1)


# Removing these columns that are descriptions of genes and not the actual expression values
cols = ["Gene name d", "Entrez gene id e", "Chromosome f", "Start f", "End f", "Cytoband f"]
exp_clean = exp.drop(columns=cols)
# Remove the prefix part (before the :) so the names match
exp_clean.columns = [c.split(":")[1].strip() for c in exp_clean.columns]

# Intersection between cell lines and expression
common = gi.index.intersection(exp_clean.columns)
print(f"Common cell lines: {len(common)}")

# New table with only intersection values
gi_final = gi.loc[common]
# X = the matching cell line columns flipped
# So rows = samples instead of genes
X = exp_clean[common].T
# Reorder cell line rows to match GI50 order
X = X.reindex(gi_final.index)
# Convert values to float; if can't convert = NaN
X = X.apply(pd.to_numeric, errors="coerce")
# Replace NaN with median value of gene
# Dropping NaN columns would harm model
X = X.fillna(X.median())

print(f"Alignment check: {all(X.index == gi_final.index)}")

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

# Map genes
probeset_to_gene = exp["Gene name d"].to_dict()
gene_names = [probeset_to_gene.get(pid, pid) for pid in X.columns.tolist()]

# Save values
np.save("X_values.npy", X.values)
np.save("X_pca.npy", X_pca)
np.save("valid_drugs.npy", np.array(valid_drugs))
np.save("sample_drugs.npy", np.array(sample_drugs))
np.save("gene_names.npy", np.array(gene_names))
gi_final.to_csv("gi_final.csv")
pd.DataFrame({"cell_line": gi_final.index}).to_csv("cell_lines.csv", index=False)

print("Saved: X_values.npy, X_pca.npy, gi_final.csv, valid_drugs.npy, sample_drugs.npy, gene_names.npy")
