<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/quipcell.svg?branch=main)](https://cirrus-ci.com/github/<USER>/quipcell)
[![ReadTheDocs](https://readthedocs.org/projects/quipcell/badge/?version=latest)](https://quipcell.readthedocs.io/en/stable/)
[![PyPI-Server](https://img.shields.io/pypi/v/quipcell.svg)](https://pypi.org/project/quipcell/)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

# quipcell

> Fine-scale cellular deconvolution based on Generalized Cross Entropy

A method to perform cellular deconvolution at a fine-scale
(i.e. single-cell or neighborhood level), using a generalization of
maximum entropy that is also an efficient convex optimization problem.

## Installation

Installation

```
pip install .
```

## Method

For a description of the method, see the
[manuscript](***REMOVED***).

## Usage

The [vignette](vignette/vignette.ipynb) provides a detailed example
applying quipcell to the Human Lung Cell Atlas.

The snippet below demonstrates how to obtain weights from two
AnnData's containing the single cell reference and the bulk samples to
deconvolve. Both AnnDatas are assumed to contain the raw counts, and
to have the same genes.

```{python}
import scanpy as sc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# save the number of UMIs per cell
adata_ref.obs['n_umi'] = adata_ref.X.sum(axis=1)

# Normalize the single cell reference
normalize_sum = 1e3
sc.pp.normalize_total(adata_ref, target_sum=normalize_sum)

# Compute low dimensional features via PCA and LDA on the single cells
sc.pp.pca(adata_ref, n_comps=100)

lda = LinearDiscriminantAnalysis(n_components=15)
lda.fit(adata_ref.obsm['X_pca'], adata_ref.obs['celltype'])

adata_ref.obsm['X_lda'] = lda.transform(adata_ref.obsm['X_pca'])

# Normalize the bulk samples
sc.pp.normalize_total(adata_bulk, target_sum=normalize_sum)

# Apply PCA rotation to pseudobulks
X = adata_bulk.X - np.squeeze(np.asarray(adata_ref.X.mean(axis=0)))
X = np.asarray(X @ adata_ref.varm['PCs'])
adata_bulk.obsm['X_pca'] = X
# Apply LDA rotation to pseudobulks
adata_bulk.obsm['X_lda'] = lda.transform(X)

# Compute the weights. Rows are reference single cells, columns are bulk samples
w_reads = qpc.estimate_weights_multisample(adata_ref.obsm['X_lda'],
                                         adata_bulk.obsm['X_lda'])

# Optional: convert read-level probabilities (probability of sampling
# a read from a neighborhood) to cell-level probabilities (probability
# of sampling a cell from a neighborhood)

# Estimate size factors using Poisson regression
size_factors = qpc.estimate_size_factors(
    adata_ref.obsm['X_lda'],
    adata_ref.obs['n_umi'].values,
    adata_ref.obs['sample'].values,
    #verbose=True
)

# Convert read-level weights to cell-level weights
w_cell = qpc.renormalize_weights(w_reads, size_factors)
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
