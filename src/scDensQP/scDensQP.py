import logging

import numpy as np
import cvxpy as cp

import scipy.sparse

logger = logging.getLogger(__name__)

def estimate_weights_multisample(X, mu_multisample,
                                 renormalize=True,
                                 **kwargs):
    w_hat_multisample = []

    for i in range(mu_multisample.shape[0]):
        prob = estimate_weights(X, mu_multisample[i,:],
                                **kwargs)

        w_hat, = prob.variables()
        w_hat = w_hat.value.copy()

        # Reference for problem statuses:
        # https://www.cvxpy.org/tutorial/intro/index.html#other-problem-statuses
        logger.info(f"i={i}, objective={prob.value}, {prob.status}")

        w_hat_multisample.append(w_hat)

    ret = np.array(w_hat_multisample).T
    if renormalize:
        ret[ret < 0] = 0
        ret = np.einsum("ij,j->ij", ret, 1.0 / ret.sum(axis=0))

    return ret

def estimate_weights(X, mu, quad_form=True, solve_kwargs=None):
    # Reference for solver options:
    # https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options
    if not solve_kwargs:
        solve_kwargs = {}
    solve_kwargs.setdefault("verbose", False)

    n = X.shape[0]
    z = np.zeros(n)

    w = cp.Variable(n)
    Xt = X.T

    if quad_form:
        objective = cp.Minimize(cp.sum_squares(w))
    else:
        # More efficient for conic solvers like ECOS or SCS, but
        # prevents using QP solvers like OSQP. See also:
        # http://cvxr.com/cvx/doc/advanced.html#eliminating-quadratic-forms
        objective = cp.Minimize(cp.norm(w, 2))

    prob = cp.Problem(
        objective,
        [w >= z,
         Xt @ w == mu,
         cp.sum(w) == 1.0]
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

# TODO: parametrize the cvxpy.Problem by mu to reduce compilation times?
# https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming

# TODO: Add accelerated projected gradient descent solver? E.g. see:
# https://stackoverflow.com/questions/65526377/cvxpy-returns-infeasible-inaccurate-on-quadratic-programming-optimization-proble
