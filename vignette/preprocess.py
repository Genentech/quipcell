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

obs_pseudobulk = (adata.obs[['sample', 'donor_id', 'study']]
      .drop_duplicates()
      .reset_index(drop=True))

assert set(obs_pseudobulk['sample']) == set(onehot.columns)

obs_pseudobulk['sample'] = obs_pseudobulk['sample'].astype(str)
obs_pseudobulk = obs_pseudobulk.set_index('sample')
obs_pseudobulk = obs_pseudobulk.loc[onehot.columns]

adata_pseudobulk = ad.AnnData(
    X=pseudobulks,
    obs=obs_pseudobulk,
    var=adata.var
)

logging.info("Saving pseudobulks")
adata_pseudobulk.write_h5ad("data/pseudobulks.h5ad")

# Save number of UMIs before renormalizing/subsetting
adata.obs['n_umi'] = adata.raw.X.sum(axis=1)

# Select highly variable genes
logging.info("Selecting highly variable genes")

sc.pp.highly_variable_genes(
    adata,
    batch_key='study'
)

# Reduce memory usage to enable analysis on laptop: Delete log counts,
# and filter to highly variable genes only
adata.X = adata.raw.X
del adata.raw

adata = adata[:,adata.var['highly_variable']]

# Adjust celltype annotations; at each level, replace "None" with the
# next level up
logging.info("Adjust celltype annotations (removing Nones)")

adata.obs['celltype_lvl1'] = adata.obs['ann_level_1']

for i in range(2,6):
    ann_finest = adata.obs[f'ann_level_{i}'].astype(str)
    is_none = (ann_finest == 'None')
    ann_finest[is_none] = adata.obs[f'celltype_lvl{i-1}'][is_none]
    adata.obs[f'celltype_lvl{i}'] = ann_finest

assert (adata.obs['ann_finest_level'] == adata.obs['celltype_lvl5']).all()

# save the modified anndata
logging.info("Saving modified anndata")
adata.write_h5ad("data/hlca_hvgs.h5ad")

# Make a dataframe of number of UMIs, cells from each celltype
logging.info("Make dataframe of celltype abundances")

df_abundance = []

for i in range(1, 6):
    df = adata.obs.rename(columns={f'celltype_lvl{i}': 'celltype'})

    df_agg = (df[['sample', 'celltype', 'n_umi']]
    .groupby(['sample', 'celltype'], observed=False)
    .sum()
    .reset_index())

    df_agg['n_umi'] = df_agg['n_umi'].astype(int)

    umi_denom = (df_agg[['sample', 'n_umi']]
             .groupby('sample', observed=False)
             .sum()
             .loc[df_agg['sample']]['n_umi']
             .values)

    df_agg['frac_umi'] = df_agg['n_umi'] / umi_denom

    n_cells = (df[['sample', 'celltype']]
    .groupby(['sample', 'celltype'], observed=False)
    .size())

    df_agg['n_cell'] = (n_cells
                        .loc[zip(df_agg['sample'], df_agg['celltype'])]
                        .values)

    cell_denom = (df_agg[['sample', 'n_cell']]
             .groupby('sample', observed=False)
             .sum()
             .loc[df_agg['sample']]['n_cell']
             .values)

    df_agg['frac_cell'] = df_agg['n_cell'] / cell_denom

    df_agg.insert(1, 'ann_level', f'lvl{i}')
    df_abundance.append(df_agg)

df_abundance = pd.concat(df_abundance)

df_abundance.to_csv("data/abundances.csv", index=False)
