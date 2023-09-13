
import pandas as pd
import anndata as ad
import scanpy as sc

# TODO Use logging.info() instead of print()
print("Loading HLCA")

# Takes 17-18 GB RAM
adata = ad.read_h5ad("data/hlca_orig.h5ad")

# First, save the number of UMIs before filtering

adata.obs['n_umi'] = adata.raw.X.sum(axis=1)

# Select highly variable genes

print("Selecting highly variable genes")

sc.pp.highly_variable_genes(
    adata,
    batch_key='study'
)

# Revert to raw counts

adata.X = adata.raw.X
del adata.raw

# save a smaller copy with just the highly variable genes

print("Saving smaller copy of dataset")

adata_hvgs = adata[:,adata.var['highly_variable']]
adata_hvgs.write_h5ad("data/hlca_hvgs.h5ad")

# Generate & save pseudobulks

print("Generating pseudobulks")

onehot = pd.get_dummies(adata.obs['sample'], dtype=float, sparse=True)
onehot_mat = onehot.sparse.to_coo().tocsr()

pseudobulks = onehot_mat.T @ adata.X

adata_pseudobulk = ad.AnnData(
    X=pseudobulks,
    obs=pd.DataFrame(index=onehot.columns),
    var=adata.var
)

print("Saving pseudobulks")

adata_pseudobulk.write_h5ad("data/pseudobulks.h5ad")
