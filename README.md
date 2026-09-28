# Drug Classification & Sensitivity Prediction

A machine learning pipeline on the NCI-60 cancer cell-line panel that predicts
drug response from gene expression, clusters drugs by potency profile, and
uses SHAP to explain which genes drive sensitivity predictions.

## Data

Three raw NCI-60 datasets are combined:

| File | Contents |
|---|---|
| `RNA__Affy_HG_U133_Plus_2.0_RMA.xls` | Gene expression (Affymetrix HG-U133 Plus 2.0, RMA-normalized) — 54,609 probes × cell line |
| `GI50.csv` | Drug potency screening data — GI50 (log10 molar concentration for 50% growth inhibition) per drug per cell line |
| `NCI60_CELL_LINE_METADATA.xls` | Cell line metadata — tissue of origin, histology, patient info |

Expression and GI50 are aligned on their common set of cell lines (59), giving
two row-matched matrices: a 59 × 54,609 expression matrix (features) and a
59 × drugs GI50 matrix (targets). Data files are tracked via [Git LFS](https://git-lfs.com/).

## Pipeline

```
data_prep.py   → clean, align, and filter raw data
models.py      → train models, run all three tasks
visualizations.py → build the results dashboard
```

### 1. Data preparation (`data_prep.py`)

- Reshapes GI50 from long to wide format (cell line × drug)
- Aligns expression and GI50 on their common 59 cell lines, in matching row order
- Fills missing expression values with the gene median; leaves GI50 gaps as-is (untested, not zero)
- Filters to drugs with ≥55 non-missing GI50 values and meaningful variance (std ≥ 0.2), then samples 100 for modeling
- Saves cleaned arrays (`X_values.npy`, `X_pca.npy`, `gi_final.csv`, `valid_drugs.npy`, `sample_drugs.npy`, `gene_names.npy`)

### 2. Modeling (`models.py`)

For each of the 100 sampled drugs, the top 100 genes most correlated with
that drug's GI50 are selected first (feature selection against 54,000+ genes
with only 59 samples), then:

- **Task 1 — Response prediction (regression):** `RandomForestRegressor`, 5-fold CV, scored by R². **Median R² = 0.35**
- **Task 2 — Drug clustering:** K-Means over each drug's GI50 profile across all cell lines, k selected via silhouette score (k=2), splitting drugs into potent vs. weak groups
- **Task 3 — Sensitivity classification:** cell lines split into sensitive/resistant at the median GI50, `RandomForestClassifier`, 5-fold CV, scored by ROC-AUC. **Median AUC = 0.83**
- **SHAP:** for the top 5 drugs by AUC, `TreeExplainer` identifies which genes drive sensitivity predictions (`shap_drug_<id>.png`)

### 3. Visualization (`visualizations.py`)

Builds `results.png`, a 2×2 dashboard: cell lines by tissue type (PCA),
R² distribution, drug clusters, and AUC distribution.

## Results

- 21,408 of 55,899 screened drugs passed the data-quality filter
- Median R² of 0.35 across 100 sampled drugs (regression)
- Median ROC-AUC of 0.83 across 100 sampled drugs (classification)
- Drugs cluster into two potency profiles (weak vs. strong responders) across the cell line panel

## Usage

```bash
python data_prep.py
python models.py
python visualizations.py
```

## Stack

Python, pandas, NumPy, scikit-learn (Random Forest, PCA, K-Means), SHAP, matplotlib/seaborn
