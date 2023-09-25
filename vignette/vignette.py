# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: quickbeam
#     language: python
#     name: quickbeam
# ---

# %%
from collections import Counter

import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import pandas as pd

import plotnine as gg
import mizani
import mizani.transforms as scales

import anndata as ad
import scanpy as sc

import scipy.sparse

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder

import quickbeam as qb
import cvxpy as cp

# %%
adata = ad.read_h5ad("data/hlca_hvgs.h5ad")


# %%
# Normalize
normalize_sum = 1000
adata.X = scipy.sparse.diags(normalize_sum / adata.obs['n_umi'].values) @ adata.X

# %%
validation_study = 'Krasnow_2020'

rng = np.random.default_rng(12345)
n = adata.obs.shape[0]

keep = adata.obs['study'] != validation_study
keep = keep & (rng.random(size=n) < .125)

adata_ref = adata[keep,:]

# %%
adata_ref

# %%
sc.pp.pca(adata_ref, n_comps=100)

# %%
# since the PCA is zero-centered, we need to save the gene-wise means
# to apply PCA rotation to pseudobulks
gene_means = adata_ref.X.mean(axis=0)
gene_means = np.squeeze(np.asarray(gene_means))
adata_ref.var['x_mean'] = gene_means

# %%
lda = LinearDiscriminantAnalysis(n_components=15)

X = adata_ref.obsm['X_pca']
y = adata_ref.obs['ann_finest_level']

lda.fit(X, y)

adata_ref.obsm['X_lda'] = lda.transform(adata_ref.obsm['X_pca'])

# %%
adata_pseudobulk = ad.read_h5ad("data/pseudobulks.h5ad")

# %%
adata_pseudobulk.raw = adata_pseudobulk
sc.pp.normalize_total(adata_pseudobulk, target_sum=normalize_sum)

# %%
adata_pseudobulk = adata_pseudobulk[:, adata.var.index]

# %%
X = adata_pseudobulk.X - adata_ref.var['x_mean'].values
X = np.asarray(X @ adata_ref.varm['PCs'])
adata_pseudobulk.obsm['X_pca'] = X
adata_pseudobulk.obsm['X_lda'] = lda.transform(X)

# %%
keep = adata_pseudobulk.obs['study'] == validation_study
adata_pseudobulk = adata_pseudobulk[keep,:]


# %%
w_umi = qb.estimate_weights_multisample(adata_ref.obsm['X_lda'],
                                        adata_pseudobulk.obsm['X_lda'])

# %%
size_factors = qb.estimate_size_factors(
    adata_ref.obsm['X_lda'],
    adata_ref.obs['n_umi'].values,
    adata_ref.obs['sample'].values,
    #verbose=True
)

w_cell = qb.renormalize_weights(w_umi, size_factors)

# %%
# Dataframe for plotting UMI-level weights on UMAP
df = pd.DataFrame(
    np.hstack([adata_ref.obsm['X_umap'], w_umi]),
    columns = ['UMAP1', 'UMAP2'] + list(adata_pseudobulk.obs.index)
).melt( # pivot to long format
    id_vars=['UMAP1', 'UMAP2'],  
    var_name='sample', 
    value_name='weight'
)

# set very small weights to 0 for prettier plotting
weight_trunc = df['weight'].values.copy()
weight_trunc[weight_trunc < 1e-9] = 0
df['weight_trunc'] = weight_trunc

(gg.ggplot(df, gg.aes(x="UMAP1", y="UMAP2", color="weight_trunc")) +
    gg.geom_point(size=.25, alpha=.5) +
    gg.facet_wrap("~sample") +
    gg.scale_color_cmap(trans=scales.log_trans(base=10)))

 # %%
 # Dataframe for plotting cell-level weights on UMAP
df = pd.DataFrame(
    np.hstack([adata_ref.obsm['X_umap'], w_cell]),
    columns = ['UMAP1', 'UMAP2'] + list(adata_pseudobulk.obs.index)
).melt( # pivot to long format
    id_vars=['UMAP1', 'UMAP2'],  
    var_name='sample', 
    value_name='weight'
)

# set very small weights to 0 for prettier plotting
weight_trunc = df['weight'].values.copy()
weight_trunc[weight_trunc < 1e-9] = 0
df['weight_trunc'] = weight_trunc

(gg.ggplot(df, gg.aes(x="UMAP1", y="UMAP2", color="weight_trunc")) +
    gg.geom_point(size=.25, alpha=.5) +
    gg.facet_wrap("~sample") +
    gg.scale_color_cmap(trans=scales.log_trans(base=10)))

# %%
adata_ref.obs['size_factor'] = size_factors
adata_ref.obs['size_factor_log10'] = np.log10(size_factors)

sc.pl.umap(adata_ref, color=['size_factor', 'size_factor_log10'])

# %%
df_abundance = pd.read_csv('data/abundances.csv')
df_abundance = df_abundance[df_abundance['sample'].isin(adata_pseudobulk.obs.index)]
df_abundance['sample'] = df_abundance['sample'].astype(str)
df_abundance

# %%
est_frac_umi = []
est_frac_cells = []

# TODO use a more descriptive variable name than i
for i in range(1, 6):
    enc = OneHotEncoder()
    mat_onehot = enc.fit_transform(
        adata_ref.obs[f'celltype_lvl{i}'].values.reshape(-1,1)
    )

    # Function to sum weights by celltype, then pivot to long dataframe
    def aggregate_and_reshape(w):
        return pd.DataFrame(
            w.T @ mat_onehot,
            columns = enc.categories_[0],
            index = adata_pseudobulk.obs.index
        ).reset_index(names='sample').melt(
            id_vars=['sample'], var_name=['celltype'], value_name='weight'
        ).assign(ann_level=f'lvl{i}')

    est_frac_umi.append(aggregate_and_reshape(w_umi))
    est_frac_cells.append(aggregate_and_reshape(w_cell))


# Function to combine the summed weights across annotation levels,
# returning a vector to be added to df_abundance
def concat_aggregated_weights(agg_weight_list):
    idx_keys = ['ann_level', 'sample', 'celltype']
    return (pd.concat(agg_weight_list)
            .set_index(idx_keys)
            .loc[zip(*[df_abundance[k] for k in idx_keys])]
            .values)

df_abundance['est_frac_umi'] = concat_aggregated_weights(est_frac_umi)
df_abundance['est_frac_cell'] = concat_aggregated_weights(est_frac_cells)

df_abundance

 # %%
 (gg.ggplot(df_abundance.sample(frac=1, random_state=42), 
           gg.aes(x="frac_umi", y="est_frac_umi", color="ann_level", shape="sample")) +
    gg.geom_point() +
    gg.geom_abline(linetype='dashed') +
    gg.scale_x_sqrt(breaks=[.01,.1,.2,.4,.6,.8]) +
    gg.scale_y_sqrt(breaks=[.01,.1,.2,.4,.6,.8]) +
    gg.theme_bw(base_size=16))

 # %%
 (gg.ggplot(df_abundance.sample(frac=1, random_state=42), 
           gg.aes(x="frac_cell", y="est_frac_cell", color="ann_level", shape="sample")) +
    gg.geom_point() +
    gg.geom_abline(linetype='dashed') +
    gg.scale_x_sqrt(breaks=[.01,.1,.2,.4,.6,.8]) +
    gg.scale_y_sqrt(breaks=[.01,.1,.2,.4,.6,.8]) +
    gg.theme_bw(base_size=16))

# %%
# TODO Add explanatory text (e.g. about the alveolar macrophages)
