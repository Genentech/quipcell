#!/usr/bin/env python

import numpy as np
import pandas as pd

import anndata as ad
import scanpy as sc

import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s")

logging.info("Loading HLCA")

# Takes 17-18 GB RAM
adata = ad.read_h5ad("data/hlca_orig.h5ad")

# Generate & save pseudobulks

logging.info("Generating pseudobulks")

onehot = pd.get_dummies(adata.obs['sample'], dtype=float, sparse=True)
onehot_mat = onehot.sparse.to_coo().tocsr()

pseudobulks = onehot_mat.T @ adata.raw.X

adata_pseudobulk = ad.AnnData(
    X=pseudobulks,
    obs=pd.DataFrame(index=onehot.columns),
    var=adata.var
)

logging.info("Saving pseudobulks")

adata_pseudobulk.write_h5ad("data/pseudobulks.h5ad")

# TODO Add study, subject_ID to adata_pseudobulk.obs

# Select highly variable genes

logging.info("Selecting highly variable genes")

sc.pp.highly_variable_genes(
    adata,
    batch_key='study'
)

# Save number of UMIs before renormalizing/subsetting
adata.obs['n_umi'] = adata.raw.X.sum(axis=1)

logging.info("Renormalizing counts")

# Delete the log-counts, switching to raw counts
adata.X = adata.raw.X
del adata.raw

# Convert to counts per 5k
target = 5000
sc.pp.normalize_total(adata, target_sum=target)
assert np.allclose(adata.X.sum(axis=1), target)
adata.uns['normalize_total_target_sum'] = target

# save a smaller copy with just the highly variable genes

logging.info("Saving smaller copy of dataset")

adata = adata[:,adata.var['highly_variable']]
adata.write_h5ad("data/hlca_hvgs.h5ad")
