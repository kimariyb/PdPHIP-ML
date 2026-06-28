# PdPHIP-ML

Machine-learning workflows for Pd-catalyzed parahydrogen-induced polarization (Pd-PHIP) alkyne screening. The repository contains notebook-based pipelines for candidate alkyne library construction, descriptor-space clustering, and supervised modeling of PHIP spin-enhancement behavior.

## Key Features

- Filter large PubChem/ZINC SMILES sources into PHIP-relevant terminal/internal alkyne candidates.
- Generate and analyze multiple molecular feature spaces: RDKit 2D descriptors, Mordred descriptors, and AQME/xTB descriptors.
- Use PCA, UMAP, hierarchical clustering, and silhouette scores to select diverse candidate molecules.
- Train classification models for high/low spin-enhancement labels.
- Train regression models for continuous PHIP spin-enhancement response.
- Compare classical ML models with leave-one-out cross-validation (LOOCV), feature filtering, and model-summary exports.

## Table of Contents

- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Data Layout](#data-layout)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Workflow](#workflow)
- [Available Scripts and Notebooks](#available-scripts-and-notebooks)
- [Expected Outputs](#expected-outputs)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Project Structure

```text
.
|-- README.md
|-- LICENSE
|-- supervised/
|   |-- classify.ipynb              # Binary classification of PHIP response
|   `-- regressor.ipynb             # Continuous-response regression models
`-- unsupervised/
    |-- filter/
    |   |-- alkynes_pubchem_fliter.py # PubChem alkyne filtering pipeline
    |   `-- alkynes_zinc_fliter.py    # ZINC alkyne filtering pipeline
    |-- RDKIT/
    |   |-- alkynes_smi.csv
    |   `-- umap_cluster_rdkit.ipynb
    |-- Mordred/
    |   |-- alkynes_smi.csv
    |   |-- preprocess_mordred.ipynb
    |   `-- umap_cluster_mordred.ipynb
    `-- AQME/
        |-- alkynes_smi.csv
        `-- umap_cluster_dft.ipynb
```

## Tech Stack

- **Language**: Python
- **Primary interface**: Jupyter Notebook
- **Cheminformatics**: RDKit, scikit-fingerprints, AQME/xTB, Mordred descriptors
- **Machine learning**: scikit-learn, PyMC, ArviZ, statsmodels
- **Dimensionality reduction**: PCA, UMAP
- **Clustering**: SciPy hierarchical clustering
- **Data analysis**: pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

The notebooks were created with Python kernels named `chem` / `Python 3`. The supervised notebooks record Python 3.10.18 metadata.

## Data Layout

### Included Data

The unsupervised descriptor notebooks include candidate alkyne SMILES files:

```text
unsupervised/RDKIT/alkynes_smi.csv
unsupervised/Mordred/alkynes_smi.csv
unsupervised/AQME/alkynes_smi.csv
```

Each file has the format:

```csv
SMILES,code_name
C#CC(=O)C(C)C,1
C#CC(=O)C(OCC)OCC,2
```

The included files currently contain 1,312 candidate molecules plus the header row.

### External or Generated Data

The supervised notebooks expect feature tables under `supervised/data/`:

```text
supervised/data/PHIP_Spin_Features.csv
supervised/data/PHIP_QSAR_Features.csv
supervised/data/PHIP_DFT_Features.csv
```

These files are not currently committed in this repository. They must be supplied before running:

```text
supervised/classify.ipynb
supervised/regressor.ipynb
```

The AQME notebook expects an AQME/xTB descriptor table:

```text
unsupervised/AQME/AQME_full_alkynes_smi.csv
```

This file is generated externally from `unsupervised/AQME/alkynes_smi.csv` using AQME `qdescp` with xTB descriptors.

## Installation

### 1. Clone the Repository

```bash
git clone git@github.com:kimariyb/PdPHIP-ML.git
cd PdPHIP-ML
```

### 2. Create a Python Environment

RDKit is easiest to install with conda or mamba:

```bash
conda create -n pdphip-ml python=3.10 -c conda-forge
conda activate pdphip-ml
```

or:

```bash
mamba create -n pdphip-ml python=3.10 -c conda-forge
mamba activate pdphip-ml
```

### 3. Install Dependencies

```bash
conda install -c conda-forge \
  jupyterlab notebook ipykernel \
  numpy pandas scipy scikit-learn statsmodels \
  matplotlib seaborn tqdm \
  rdkit umap-learn pymc arviz
```

Install packages that may not be available in your conda setup:

```bash
pip install scikit-fingerprints aqme
```

Register the kernel:

```bash
python -m ipykernel install --user --name pdphip-ml --display-name "PdPHIP-ML"
```

### 4. Optional System Dependencies

For AQME/xTB descriptor generation, install xTB and confirm it is available:

```bash
xtb --version
```

If `xtb` is missing, install it through conda-forge:

```bash
conda install -c conda-forge xtb
```

## Quick Start

Start Jupyter:

```bash
jupyter lab
```

Recommended first notebooks:

```text
unsupervised/RDKIT/umap_cluster_rdkit.ipynb
unsupervised/Mordred/umap_cluster_mordred.ipynb
```

For supervised modeling, first place the required feature files in:

```text
supervised/data/
```

Then run:

```text
supervised/classify.ipynb
supervised/regressor.ipynb
```

## Workflow

### 1. Candidate Alkyne Filtering

The filter scripts are used to screen large molecular databases for alkyne candidates that match project-specific structural constraints.

#### ZINC Filtering

`unsupervised/filter/alkynes_zinc_fliter.py` expects a tab-separated input file:

```text
SMILES<TAB>ZINC_ID
```

Run:

```bash
python unsupervised/filter/alkynes_zinc_fliter.py \
  --input /path/to/ZINC_dataset.tsv \
  --output filtered_unique_smiles.txt
```

The script:

- Parses SMILES with RDKit.
- Keeps neutral organic molecules with allowed elements: C, H, N, O, F, Cl, Br.
- Applies molecular-weight, carbon-count, heavy-atom-count, and logP filters.
- Requires an alkyne motif.
- Excludes symmetric alkynes by default.
- Deduplicates structures using InChIKey.
- Adds a small set of experimental alkyne SMILES before final deduplication.

#### PubChem Filtering

`unsupervised/filter/alkynes_pubchem_fliter.py` is configured for large PubChem CID-SMILES tables and uses multiprocessing. In its current form, input/output paths are hard-coded near the bottom of the file:

```python
params = {
    "in_path": "/mnt/data/kimariyb/dataset/pubchem/CID-SMILES.csv",
    "out_path": "/mnt/data/kimariyb/dataset/pubchem/alkynes_cid_smi.csv",
    "nproc": None,
    "csv_chunksize": 100000,
    "pool_chunksize": 5000,
}
```

Edit those paths before running:

```bash
python unsupervised/filter/alkynes_pubchem_fliter.py
```

### 2. Unsupervised Descriptor-Space Analysis

The unsupervised notebooks map candidate alkynes into descriptor spaces, reduce dimensionality, cluster molecules, and select representative candidates.

#### RDKit Descriptor Workflow

Notebook:

```text
unsupervised/RDKIT/umap_cluster_rdkit.ipynb
```

Input:

```text
unsupervised/RDKIT/alkynes_smi.csv
```

Main steps:

- Convert SMILES to RDKit molecules.
- Generate RDKit 2D descriptor fingerprints with `RDKit2DDescriptorsFingerprint`.
- Standardize features.
- Remove zero-variance and highly correlated descriptors.
- Run PCA and UMAP.
- Evaluate hierarchical clustering with silhouette scores.

#### Mordred Descriptor Workflow

Notebooks:

```text
unsupervised/Mordred/preprocess_mordred.ipynb
unsupervised/Mordred/umap_cluster_mordred.ipynb
```

Input:

```text
unsupervised/Mordred/alkynes_smi.csv
```

Main steps:

- Generate Mordred-style descriptor fingerprints with `MordredFingerprint`.
- Remove descriptors containing null values.
- Standardize and filter descriptors.
- Perform PCA, UMAP, and hierarchical clustering.

#### AQME/xTB Descriptor Workflow

Notebook:

```text
unsupervised/AQME/umap_cluster_dft.ipynb
```

Input:

```text
unsupervised/AQME/alkynes_smi.csv
```

Generated input expected by the clustering cells:

```text
unsupervised/AQME/AQME_full_alkynes_smi.csv
```

The notebook includes an AQME example:

```python
from aqme.qdescp import qdescp

qdescp(
    program="xtb",
    input="alkynes_smi.csv",
    qdescp_atoms=["C#C"],
    nproc=63,
)
```

Adjust `nproc` to match your machine.

### 3. Supervised Classification

Notebook:

```text
supervised/classify.ipynb
```

Required inputs:

```text
supervised/data/PHIP_Spin_Features.csv
supervised/data/PHIP_QSAR_Features.csv
supervised/data/PHIP_DFT_Features.csv
```

Target:

```python
RESPONSE_LABEL = "SNE"
y = spin_df[RESPONSE_LABEL].apply(lambda x: 0 if x < 100 else 1)
```

Main steps:

- Merge spin, QSAR, and DFT features by `LABEL`.
- Remove low-variance features.
- Standardize descriptors.
- Remove highly correlated features using Spearman correlation.
- Use Mann-Whitney U tests for feature filtering.
- Train/evaluate binary classifiers with LOOCV.
- Fit Bayesian logistic regression with PyMC for posterior interpretation.
- Compare models including logistic regression, SVC, random forest, and decision tree.

### 4. Supervised Regression

Notebook:

```text
supervised/regressor.ipynb
```

Required inputs:

```text
supervised/data/PHIP_Spin_Features.csv
supervised/data/PHIP_QSAR_Features.csv
supervised/data/PHIP_DFT_Features.csv
```

Target:

```python
RESPONSE_LABEL = "SNE"
y_df = spin_df[RESPONSE_LABEL].apply(lambda x: np.log(x) if x > 0 else 0)
```

Main steps:

- Merge feature tables by `LABEL`.
- Apply variance and correlation filtering.
- Screen single features with OLS and R2 thresholds.
- Search 2- and 3-variable MLR equations.
- Evaluate models with LOOCV.
- Compare MLR, LASSO, Ridge, Elastic Net, Bayesian Ridge, SVR, PLS, decision tree, and random forest.
- Export model summaries and MLR search results.

## Available Scripts and Notebooks

| Path | Purpose |
| --- | --- |
| `unsupervised/filter/alkynes_zinc_fliter.py` | Filter ZINC SMILES and write unique canonical alkyne candidates. |
| `unsupervised/filter/alkynes_pubchem_fliter.py` | Multiprocessing PubChem CID-SMILES filtering pipeline. |
| `unsupervised/RDKIT/umap_cluster_rdkit.ipynb` | RDKit descriptor generation, UMAP/PCA projection, and clustering. |
| `unsupervised/Mordred/preprocess_mordred.ipynb` | Mordred descriptor preprocessing. |
| `unsupervised/Mordred/umap_cluster_mordred.ipynb` | Mordred descriptor UMAP/PCA projection and clustering. |
| `unsupervised/AQME/umap_cluster_dft.ipynb` | AQME/xTB descriptor projection and clustering. |
| `supervised/classify.ipynb` | Binary classification for PHIP response. |
| `supervised/regressor.ipynb` | Continuous-response regression for PHIP response. |

## Expected Outputs

Depending on which notebook cells are executed, the project can generate:

- UMAP/PCA plots for candidate alkyne libraries.
- Hierarchical clustering dendrograms.
- Representative SMILES selected per cluster.
- Classification metrics such as accuracy, ROC-AUC, confusion matrix, and classification report.
- Regression metrics such as R2, MAE, RMSE, Pearson r, and Spearman rho.
- Regression summary tables:

```text
supervised/model_summary.csv
supervised/mlr_3var_search_results.csv
```

Some notebooks save figures to local folders such as `fig/`. Create the folder first if a save cell fails:

```bash
mkdir -p supervised/fig
```

## Reproducibility Notes

- Several models and dimensionality-reduction steps use stochastic algorithms. Set `random_state` consistently when comparing results.
- PyMC sampling in `classify.ipynb` can be slow because the notebook performs LOOCV and samples a Bayesian model inside each fold.
- Feature filtering in the notebooks is currently performed on the full dataset before LOOCV. This is useful for exploratory analysis, but strict predictive validation should move feature selection inside each training fold.
- The notebooks assume they are run from their own directory or that relative paths are adjusted accordingly. For example, `supervised/classify.ipynb` reads `./data/PHIP_Spin_Features.csv`, so Jupyter should be launched with `supervised/` as the working directory or the path should be edited.

## Troubleshooting

### `ModuleNotFoundError: No module named 'rdkit'`

Install RDKit from conda-forge:

```bash
conda install -c conda-forge rdkit
```

### `ModuleNotFoundError: No module named 'skfp'`

Install scikit-fingerprints:

```bash
pip install scikit-fingerprints
```

### `FileNotFoundError: ./data/PHIP_Spin_Features.csv`

The supervised feature tables are not committed. Add them under:

```text
supervised/data/
```

Then rerun the notebook from the `supervised/` working directory.

### `FileNotFoundError: ./AQME_full_alkynes_smi.csv`

Generate AQME/xTB descriptors first, or place the generated descriptor table under:

```text
unsupervised/AQME/AQME_full_alkynes_smi.csv
```

### xTB or AQME Fails

Check that xTB is installed and visible:

```bash
xtb --version
```

For large candidate sets, reduce `nproc` in the AQME call if the machine runs out of memory or temporary disk space.

### Notebook Path Issues

The notebooks use relative paths such as `./alkynes_smi.csv` and `./data/...`. If a file exists but cannot be found, either:

- Open Jupyter from the notebook's directory, or
- Update the path in the notebook cell.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
