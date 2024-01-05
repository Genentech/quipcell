import logging

from datetime import datetime

import numpy as np
import pandas as pd

import scipy.sparse

import cvxpy as cp

from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder

from .maxent_dual_lbfgs import estimate_weights_maxent_dual_lbfgs

logger = logging.getLogger(__name__)

# TODO: parametrize the cvxpy.Problem by mu to reduce compilation times?
# https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming
def estimate_weights_multisample(
        X, mu_multisample,
        renormalize=True,
        alpha='pearson',
        use_dual_lbfgs=False,
        **kwargs
):
    """Estimate density weights for multiple samples on a single-cell reference.

    Returns a numpy array with rows=cells, columns=samples.

    ``**kwargs`` are passed onto :meth:`quipcell.estimate_weights`.

    :param `numpy.ndarray` X: Reference embedding. Rows=cells, columns=features.
    :param `numpy.ndarray` mu_multisample: Sample moments. Either bulk gene counts (for bulk deconvolution) or sample centroids of single cells (for differential abundance). Rows=samples, columns=features.
    :param bool renormalize: Correct for any solver inaccuracies by setting small negative numbers to 0, and renormalizing sums to 1.
    :param float alpha: Value of alpha for alpha-divergence. Also accepts 'pearson' for alpha=2 (which is a quadratic program) or 'kl' for alpha=1 (which is same as maximum entropy).
    :param bool use_dual_lbfgs: If True, solve via the dual problem with L-BFGS-B, instead of using cvxpy. Only implemented for alpha==1 or alpha=='kl'.

    :rtype: :class:`numpy.ndarray`
    """
    start = datetime.now()

    w_hat_multisample = []

    n = mu_multisample.shape[0]
    for i in range(n):
        if use_dual_lbfgs and (alpha == 1 or alpha == 'kl'):
            res = estimate_weights_maxent_dual_lbfgs(
                X, mu_multisample[i,:],
                **kwargs
            )

            w_hat = res['primal']
        else:
            prob = estimate_weights(X, mu_multisample[i,:],
                                    alpha=alpha,
                                    **kwargs)

            # Reference for problem statuses:
            # https://www.cvxpy.org/tutorial/intro/index.html#other-problem-statuses
            logger.info(f"i={i}, objective={prob.value}, {prob.status}")

            w_hat, = prob.variables()
            w_hat = w_hat.value.copy()

        w_hat_multisample.append(w_hat)

    ret = np.array(w_hat_multisample).T
    if renormalize:
        ret[ret < 0] = 0
        ret = np.einsum("ij,j->ij", ret, 1.0 / ret.sum(axis=0))

    interval = (datetime.now() - start).total_seconds()
    logger.info(f"Finished {n} samples in {interval} seconds.")

    return ret

# TODO: Add accelerated projected gradient descent solver? E.g. see:
# https://stackoverflow.com/questions/65526377/cvxpy-returns-infeasible-inaccurate-on-quadratic-programming-optimization-proble

def estimate_weights(X, mu, use_norm=False, alpha='pearson',
                     relax_moment_condition=0, solve_kwargs=None):
    """Estimate density weights for a single sample on a single-cell reference.

    :param `numpy.ndarray` X: Reference embedding. Rows=cells, columns=features.
    :param `numpy.ndarray` mu: Sample moments. Either bulk gene counts (for bulk deconvolution) or sample centroids of single cells (for differential abundance). Should be a 1-dimensional array.
    :param bool use_norm: Whether to optimize the pnorm sum(w**alpha)**(1/alpha) instead of sum(w**alpha). While mathematically equivalent when alpha > 1, the conditioning of the optimization problem may be better with pnorm objective. However, it prevents using efficient quadratic optimization solvers when alpha=2. See here for discussion: http://cvxr.com/cvx/doc/advanced.html#eliminating-quadratic-forms
    :param float alpha: Value of alpha for alpha-divergence. Also accepts 'pearson' for alpha=2 (which is a quadratic program) or 'kl' for alpha=1 (which is same as maximum entropy).
    :param float relax_moment_condition: For moment constraints, require solution to be within this distance of the data moments. Default is 0, which means to require the moments to match exactly.
    :param dict solve_kwargs: Additional kwargs to pass to `cvxpy.Problem.solve`.

    :rtype: :class:`cvxpy.Problem`
    """
    # Reference for solver options:
    # https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options
    if not solve_kwargs:
        solve_kwargs = {}
    solve_kwargs.setdefault("verbose", False)

    n = X.shape[0]
    z = np.zeros(n)

    w = cp.Variable(n)

    # Initialize as uniform distribution
    # NOTE: Unclear in which solvers CVXPY will actually use the initialization
    # https://www.cvxpy.org/tutorial/advanced/index.html#warm-start
    # https://stackoverflow.com/questions/52314581/initial-guess-warm-start-in-cvxpy-give-a-hint-of-the-solution
    w.value = np.ones(n) / n

    Xt = X.T

    if alpha == 'pearson':
        alpha = 2
    elif alpha == 'kl':
        alpha = 1
    elif type(alpha) == str:
        raise ValueError(f'Unrecognized divergence {alpha}')

    if use_norm:
        if alpha <= 1:
            raise ValueError('use_norm requires alpha > 1')
        objective = cp.Minimize(cp.norm(w, alpha))
    elif alpha == 1:
        objective = cp.Maximize(cp.sum(cp.entr(w)))
    elif alpha == 0:
        objective = cp.Maximize(cp.sum(cp.log(w)))
    elif alpha == 2:
        objective = cp.Minimize(cp.sum_squares(w))
    elif alpha < 1 and alpha > 0:
        # sign of alpha*(alpha-1) is negative in this case
        objective = cp.Maximize(cp.sum(w**alpha))
    else:
        objective = cp.Minimize(cp.sum(w**alpha))

    constraints = [w >= z, cp.sum(w) == 1.0]
    if relax_moment_condition == 0:
        constraints.append(
            Xt @ w == mu,
        )
    else:
        constraints.append(
            Xt @ w - mu <= relax_moment_condition
        )

        constraints.append(
            Xt @ w - mu >= -relax_moment_condition
        )

    prob = cp.Problem(
        objective,
        constraints
    )

    res = prob.solve(**solve_kwargs)
    assert prob.variables()[0] is w
    assert prob.value is res

    # TODO: Reenable assertions, with params for atol/rtol?
    #w_hat, = prob.variables()
    #w_hat = w_hat.value
    #assert np.all(np.isclose(w_hat, 0, atol=1e-4) | (w_hat >= 0))
    #assert np.allclose(np.sum(w_hat), 1)

    return prob

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
