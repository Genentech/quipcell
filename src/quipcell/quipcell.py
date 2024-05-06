import logging

from datetime import datetime

import numpy as np
import pandas as pd

import scipy.sparse

from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder

from .solvers import AlphaDivergenceCvxpySolver
from .maxent_dual_lbfgs import MaxentDualLbfgsSolver

logger = logging.getLogger(__name__)

# TODO: parametrize the cvxpy.Problem by mu to reduce compilation times?
# https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming
def estimate_weights_multisample(
        X, mu_multisample,
        renormalize=True,
        alpha='pearson',
        use_dual_lbfgs=None,
        **kwargs
):
    """Estimate density weights for multiple samples on a single-cell reference.

    Returns a numpy array with rows=cells, columns=samples.

    ``**kwargs`` are passed onto :meth:`quipcell.estimate_weights`.

    :param `numpy.ndarray` X: Reference embedding. Rows=cells, columns=features.
    :param `numpy.ndarray` mu_multisample: Sample moments. Either bulk gene counts (for bulk deconvolution) or sample centroids of single cells (for differential abundance). Rows=samples, columns=features.
    :param bool renormalize: Correct for any solver inaccuracies by setting small negative numbers to 0, and renormalizing sums to 1.
    :param float alpha: Value of alpha for alpha-divergence. Also accepts 'pearson' for alpha=2 (which is a quadratic program) or 'kl' for alpha=1 (which is same as maximum entropy).
    :param bool use_dual_lbfgs: If True, solve via the dual problem with L-BFGS-B, instead of using cvxpy. Currently only implemented for alpha==1 or alpha=='kl'. Default is True when alpha==1 or 'kl', and False otherwise.

    :rtype: :class:`numpy.ndarray`
    """

    start = datetime.now()

    if use_dual_lbfgs is None:
        use_dual_lbfgs = (alpha == 1 or alpha == 'kl')

    if not use_dual_lbfgs:
        solver = AlphaDivergenceCvxpySolver(alpha=alpha, **kwargs)
    elif alpha == 1 or alpha == 'kl':
        solver = MaxentDualLbfgsSolver(**kwargs)
    else:
        raise NotImplementedError("Dual LBFGS solver only implemented for alpha=1")

    solver.fit(X, mu_multisample)
    return solver.weights(renormalize=renormalize)

def renormalize_weights(weights, size_factors):
    """Renormalize weights to correct for cell-specific size factors.

    This can be used to convert read-level probability weights to
    cell-level weights.

    :param `numpy.ndarray` weights: Weights as returned by `quipcell.estimate_weights_multisample`. Rows=cells, columns=samples.
    :param `numpy.ndarray` size_factors: Cell-level size factors. Should be 1 dimensional, with length equal to the number of rows in `weights`.

    :rtype: :class:`numpy.ndarray`
    """
    weights = np.einsum("ij,i->ij", weights, 1/size_factors)
    weights = np.einsum("ij,j->ij", weights, 1/weights.sum(axis=0))
    assert np.allclose(weights.sum(axis=0), 1)
    return weights

def estimate_size_factors(X, n_reads, sample, **kwargs):
    """Estimate cell-level size factors with Poisson regression to control for technical variation between samples.

    Returns a vector of the size factor for each cell.

    :param `numpy.ndarray` X: Single cell embedding. Rows=cells, columns=features.
    :param `numpy.ndarray` n_reads: 1d vector of the number of reads per cell.
    :param `numpy.ndarray` sample: 1d string vector for the sample that each cell came from.

    :rtype: :class:`numpy.ndarray`
    """
    enc = OneHotEncoder()

    modmat = enc.fit_transform(sample.reshape(-1, 1))
    modmat = np.hstack([X, np.asarray(modmat.todense())])
    
    kwargs.setdefault("fit_intercept", False)
    kwargs.setdefault("alpha", 0)
    kwargs.setdefault("solver", "newton-cholesky")
    kwargs.setdefault("verbose", 0)

    clf = linear_model.PoissonRegressor(**kwargs)

    clf.fit(modmat, n_reads)

    size_factors = np.exp(X @ clf.coef_[:X.shape[1]])
    size_factors = size_factors / np.mean(size_factors)
    return size_factors
