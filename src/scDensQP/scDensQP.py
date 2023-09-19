import logging

import numpy as np
import cvxpy as cp

import scipy.sparse

logger = logging.getLogger(__name__)

def estimate_weights_multisample(X, mu_multisample,
                                 quad_form=True,
                                 **solver_kwargs):
    w_hat_multisample = []

    kwargs = {"verbose": False}
    kwargs.update(solver_kwargs)

    for i in range(mu_multisample.shape[0]):
        prob = estimate_weights(X, mu_multisample[i,:],
                                quad_form=quad_form,
                                **kwargs)

        w_hat, = prob.variables()
        w_hat = w_hat.value.copy()

        norm = prob.value #float
        status = prob.status #string
        logger.info(f"i={i}, obj={norm}, {status}")

        w_hat_multisample.append(w_hat)

    ret = np.array(w_hat_multisample).T
    ret[ret < 0] = 0
    ret = np.einsum("ij,j->ij", ret, 1.0 / ret.sum(axis=0))
    return ret

def estimate_weights(X, mu, quad_form=True, **solver_kwargs):
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

    res = prob.solve(**solver_kwargs)
    assert prob.variables()[0] is w
    assert prob.value is res

    w_hat, = prob.variables()
    w_hat = w_hat.value
    assert np.all(np.isclose(w_hat, 0) | (w_hat >= 0))
    assert np.allclose(np.sum(w_hat), 1)

    return prob

# TODO: parametrize the cvxpy.Problem by mu to reduce compilation times?
# https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming

# TODO: Add accelerated projected gradient descent solver? E.g. see:
# https://stackoverflow.com/questions/65526377/cvxpy-returns-infeasible-inaccurate-on-quadratic-programming-optimization-proble
