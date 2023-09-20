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
adata

# %%
# X should be scaled but not log-transformed. Check that multiplying
# by normalization factors yields an integer.
assert np.allclose(
    0, np.modf((scipy.sparse.diags(adata.obs['n_umi'].values) * adata.X *
                adata.uns['normalize_total_target_sum']).data)[0]
)

# TODO: Notebook would be clearer if adata just contained counts, and
# we manually normalized by n_umi, rather than doing it in
# preprocessing step

# %%
sc.pp.pca(adata, n_comps=100)

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
sc.pp.normalize_total(adata_pseudobulk, target_sum=adata.uns['normalize_total_target_sum'])

# %%
adata_pseudobulk = adata_pseudobulk[:, adata.var.index]

# %%
X = adata_pseudobulk.X - adata.var['x_mean'].values
X = np.asarray(X @ adata.varm['PCs'])
adata_pseudobulk.obsm['X_pca'] = X
adata_pseudobulk.obsm['X_lda'] = lda.transform(X)

# %%
# Add study, subject_ID to adata_pseudobulk.obs
# TODO Move this preprocess.py

df = (adata.obs[['sample', 'donor_id', 'study']]
      .drop_duplicates()
      .reset_index(drop=True))

assert set(df['sample']) == set(adata_pseudobulk.obs.index)

df['sample'] = df['sample'].astype(str)
df = df.set_index('sample')
df = df.loc[adata_pseudobulk.obs.index]

adata_pseudobulk.obs = df

# %%
keep = adata_pseudobulk.obs['study'] == validation_study
adata_pseudobulk = adata_pseudobulk[keep,:]


# %%
keep = adata.obs['cohort'] == 'reference'
#keep = adata.obs['cohort'] != 'validation'
adata_ref = adata[keep,:]
res = scdqp.estimate_weights_multisample(adata_ref.obsm['X_lda'],
                                         adata_pseudobulk.obsm['X_lda'])

# %%
# Make a dataframe for plotting the weights on UMAP
df = np.hstack(
    [adata_ref.obsm['X_umap'],
     res]
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

# %%
(gg.ggplot(df, gg.aes(x="UMAP1", y="UMAP2", color="weight")) +
    gg.geom_point(size=.25, alpha=.5) +
    gg.facet_wrap("~sample") +
    gg.scale_color_cmap(trans=scales.log_trans(base=10)))

# %%
(gg.ggplot(df, gg.aes(x="UMAP1", y="UMAP2", color="weight_trunc")) +
    gg.geom_point(size=.25, alpha=.5) +
    gg.facet_wrap("~sample") +
    gg.scale_color_cmap(trans=scales.log_trans(base=10)))

# %%

# TODO Move this to preprocess.py

adata.obs['ann_finest_1'] = adata.obs['ann_level_1']

for i in range(2,6):
    ann_finest = adata.obs[f'ann_level_{i}'].astype(str)
    is_none = (ann_finest == 'None')
    ann_finest[is_none] = adata.obs[f'ann_finest_{i-1}'][is_none]
    adata.obs[f'ann_finest_{i}'] = ann_finest

# %%
assert (adata.obs['ann_finest_level'] == adata.obs['ann_finest_5']).all()

# %%
# TODO plot cell-level weights on UMAP
# TODO plot accuracy of cell-level counts

# %%
# TODO Move this to preprocess.py
df_frac_umi = []

for i in range(1, 6):
    df = adata.obs.rename(columns={f'ann_finest_{i}': 'celltype'})
    df = (df[['sample', 'celltype', 'n_umi']]
    .groupby(['sample', 'celltype'], observed=False)
    .sum()
    .reset_index())

    denom = (df[['sample', 'n_umi']]
             .groupby('sample', observed=False)
             .sum()
             .loc[df['sample']]['n_umi']
             .values)

    df['frac_umi'] = df['n_umi'] / denom
    df['ann_level'] = f'lvl{i}'
    df_frac_umi.append(df)

df_frac_umi = pd.concat(df_frac_umi)
df_frac_umi

# %%
# Make a dataframe for plotting the weights on UMAP
df_est_frac_umi = []

for i in range(1, 6):
    df = pd.DataFrame(
        res,
        columns = adata_pseudobulk.obs.index,
        index = adata_ref.obs.index
    )

    df['celltype'] = adata_ref.obs[f'ann_finest_{i}']

    df = pd.melt(df, id_vars=['celltype'],
                var_name='sample', value_name='weight')

    df = df.groupby(['sample', 'celltype'], observed=False).sum()
    df['ann_level'] = f'lvl{i}'
    df_est_frac_umi.append(df)

df_est_frac_umi = pd.concat(df_est_frac_umi).reset_index().set_index(['ann_level', 'sample', 'celltype'])
df_est_frac_umi

# %%
df = df_frac_umi
df = df[df['sample'].isin(adata_pseudobulk.obs.index)]
df['est_frac'] = df_est_frac_umi.loc[zip(df['ann_level'], df['sample'], df['celltype'])].values
df['sample'] = df['sample'].astype(str)
df

# %%
(gg.ggplot(df.sample(frac=1), 
           gg.aes(x="frac_umi", y="est_frac", color="ann_level", shape="sample")) +
    gg.geom_point() +
    gg.geom_abline(linetype='dashed') +
    gg.scale_x_sqrt() + gg.scale_y_sqrt() +
    gg.theme_bw(base_size=16))

# %%
