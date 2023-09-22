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
#     display_name: scdensqp
#     language: python
#     name: scdensqp
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

import scDensQP as scdqp
import cvxpy as cp

# %%
adata = ad.read_h5ad("data/hlca_hvgs.h5ad")


# %%
# TODO make this less ugly
# Or just delete it?
sc.pl.umap(adata, color='ann_finest_level', legend_loc='on data')

# %%
# Normalize
normalize_sum = 1000
adata.X = scipy.sparse.diags(normalize_sum / adata.obs['n_umi'].values) @ adata.X

# TODO Just use scanpy normalize_total() instead? Even tho it doesn't
# account for the non-hvgs, it's probably fine, and would simplify exposition.
# NOTE that the order of normalization of pseudobulks would also have to change.
# Maybe should also adjust the size factor estimates to use the hvgs sum instead of total umi sum?

# %%
sc.pp.pca(adata, n_comps=100)

# TODO: Is PCA really necessary? Could we just use LDA directly on the counts?

# %%
# since the PCA is zero-centered, we need to save the gene-wise means
# to apply PCA rotation to pseudobulks
gene_means = adata.X.mean(axis=0)
gene_means = np.squeeze(np.asarray(gene_means))
adata.var['x_mean'] = gene_means

# %%
# seed from secrets.randbits(128)
rng = np.random.default_rng(293711445274383262807231515229626646634)

adata.obs['cohort'] = 'query'

n = adata.obs.shape[0]

# for speed reasons, use a smaller reference
adata.obs['cohort'][rng.random(size=n) < .125] = 'reference'

validation_study = 'Krasnow_2020'

adata.obs['cohort'][adata.obs['study'] == validation_study] = 'validation'

# TODO Simplify this by dropping the query cohort?
# Just create an adata_ref object instead, and use that for everything (e.g. LDA)

# %%
Counter(adata.obs['cohort'])

# %%
lda = LinearDiscriminantAnalysis(n_components=15)

keep = adata.obs['cohort'] != 'validation'

X = adata.obsm['X_pca'][keep,:]
y = adata.obs['ann_finest_level'][keep]

lda.fit(X, y)

adata.obsm['X_lda'] = lda.transform(adata.obsm['X_pca'])

# %%
adata_pseudobulk = ad.read_h5ad("data/pseudobulks.h5ad")

# %%
adata_pseudobulk.raw = adata_pseudobulk
sc.pp.normalize_total(adata_pseudobulk, target_sum=normalize_sum)

# %%
adata_pseudobulk = adata_pseudobulk[:, adata.var.index]

# %%
X = adata_pseudobulk.X - adata.var['x_mean'].values
X = np.asarray(X @ adata.varm['PCs'])
adata_pseudobulk.obsm['X_pca'] = X
adata_pseudobulk.obsm['X_lda'] = lda.transform(X)

# %%
keep = adata_pseudobulk.obs['study'] == validation_study
adata_pseudobulk = adata_pseudobulk[keep,:]


# %%
keep = adata.obs['cohort'] == 'reference'
#keep = adata.obs['cohort'] != 'validation'
adata_ref = adata[keep,:]
res = scdqp.estimate_weights_multisample(adata_ref.obsm['X_lda'],
                                         adata_pseudobulk.obsm['X_lda'])
# TODO Add timing here? Or maybe in the function itself

# %%
size_factors = scdqp.estimate_size_factors(
    adata_ref.obsm['X_lda'],
    adata_ref.obs['n_umi'].values,
    adata_ref.obs['sample'].values,
    #verbose=True
)
# TODO kwargs to control verbosity and other args

res_reweight = scdqp.renormalize_weights(res, size_factors)

# %%
# TODO Make a helper function for plotting weights?

# Dataframe for plotting the weights on UMAP
df = pd.DataFrame(
    np.hstack([adata_ref.obsm['X_umap'], res]),
    columns = ['UMAP1', 'UMAP2'] + list(adata_pseudobulk.obs.index)
)

# pivot to long format
df = pd.melt(df, id_vars=['UMAP1', 'UMAP2'],
             var_name='sample', value_name='weight')

# set very small weights to 0 for plotting purposes
weight_trunc = df['weight'].values.copy()
weight_trunc[weight_trunc < 1e-9] = 0
df['weight_trunc'] = weight_trunc

(gg.ggplot(df, gg.aes(x="UMAP1", y="UMAP2", color="weight_trunc")) +
    gg.geom_point(size=.25, alpha=.5) +
    gg.facet_wrap("~sample") +
    gg.scale_color_cmap(trans=scales.log_trans(base=10)))

# %%
# Make a dataframe for plotting the weights on UMAP
df = np.hstack(
    [adata_ref.obsm['X_umap'],
     res_reweight]
)

df = pd.DataFrame(
    df,
    columns = ['UMAP1', 'UMAP2'] + list(adata_pseudobulk.obs.index)
)

df = pd.melt(df, id_vars=['UMAP1', 'UMAP2'],
             var_name='sample', value_name='weight')

# set very small weights to 0 for plotting purposes
weight_trunc = df['weight'].values.copy()
weight_trunc[weight_trunc < 1e-9] = 0
df['weight_trunc'] = weight_trunc

(gg.ggplot(df, gg.aes(x="UMAP1", y="UMAP2", color="weight_trunc")) +
    gg.geom_point(size=.125, alpha=.5) +
    gg.facet_wrap("~sample") +
    gg.scale_color_cmap(trans=scales.log_trans(base=10)))

# %%
adata_ref.obs['size_factor'] = size_factors
adata_ref.obs['size_factor_log10'] = np.log10(size_factors)

sc.pl.umap(adata_ref, color=['size_factor', 'size_factor_log10'])

# %%
df_abundance = pd.read_csv('data/abundances.csv')
df_abundance

# %%
# Make a dataframe for plotting the weights on UMAP
df_est_frac_umi = []

for i in range(1, 6):
    df = pd.DataFrame(
        res,
        columns = adata_pseudobulk.obs.index,
        index = adata_ref.obs.index
    )

    df['celltype'] = adata_ref.obs[f'celltype_lvl{i}']

    df = pd.melt(df, id_vars=['celltype'],
                var_name='sample', value_name='weight')

    df = df.groupby(['sample', 'celltype'], observed=False).sum()
    df['ann_level'] = f'lvl{i}'
    df_est_frac_umi.append(df)

df_est_frac_umi = pd.concat(df_est_frac_umi).reset_index().set_index(['ann_level', 'sample', 'celltype'])
df_est_frac_umi

# %%
# Make a dataframe for plotting the weights on UMAP
df_est_frac_cells = []

for i in range(1, 6):
    df = pd.DataFrame(
        res_reweight,
        columns = adata_pseudobulk.obs.index,
        index = adata_ref.obs.index
    )

    df['celltype'] = adata_ref.obs[f'celltype_lvl{i}']

    df = pd.melt(df, id_vars=['celltype'],
                var_name='sample', value_name='weight')

    df = df.groupby(['sample', 'celltype'], observed=False).sum()
    df['ann_level'] = f'lvl{i}'
    df_est_frac_cells.append(df)

df_est_frac_cells = pd.concat(df_est_frac_cells).reset_index().set_index(['ann_level', 'sample', 'celltype'])
df_est_frac_cells

# %%
df = df_abundance
df = df[df['sample'].isin(adata_pseudobulk.obs.index)]
df['est_frac_umi'] = df_est_frac_umi.loc[zip(df['ann_level'], df['sample'], df['celltype'])].values
df['est_frac_cell'] = df_est_frac_cells.loc[zip(df['ann_level'], df['sample'], df['celltype'])].values
df['sample'] = df['sample'].astype(str)
df

# %%
(gg.ggplot(df.sample(frac=1), 
           gg.aes(x="frac_umi", y="est_frac_umi", color="ann_level", shape="sample")) +
    gg.geom_point() +
    gg.geom_abline(linetype='dashed') +
    gg.scale_x_sqrt() + gg.scale_y_sqrt() +
    gg.theme_bw(base_size=16))

# %%
(gg.ggplot(df.sample(frac=1), 
           gg.aes(x="frac_cell", y="est_frac_cell", color="ann_level", shape="sample")) +
    gg.geom_point() +
    gg.geom_abline(linetype='dashed') +
    gg.scale_x_sqrt() + gg.scale_y_sqrt() +
    gg.theme_bw(base_size=16))

# %%
# TODO adjust x,y labels of sqrt scaled plots
# TODO only show 1 umap weights plot
# TODO Add explanatory text (e.g. about the alveolar macrophages)
# TODO Use seed for pandas row shuffling
