import logging

import jax
import jax.numpy as jnp
import scipy
import scipy.optimize

from ._solvers import GeneralizedDivergenceSolver

logger = logging.getLogger(__name__)


def maxent_dual(lambd, b, A):
    return -b @ lambd - jnp.log(jnp.sum(jnp.exp(-A.T @ lambd)))


def maxent_solution(lambd_star, A):
    nu_star = jnp.log(jnp.sum(jnp.exp(-A.T @ lambd_star))) - 1
    return 1.0 / jnp.exp(A.T @ lambd_star + nu_star + 1)


def maxent_solve_dual_lbfgs(A, b, n_inequality, opt_kwargs=None):
    """Solve maximum entropy via the dual and L-BFGS-B. Based on Boyd
    & Vandenberge's Convex Optimization, Example 5.3, with the same
    notation. A minor difference difference is we allow equality
    constraints; the first n_inequality rows of A correspond to
    inequalities (a_i^T x <= b) while the remaining rows correspond to
    equalities (a_i^T x = b). opt_kwargs are passed to
    scipy.optimize.minimize. Returns a dictionary with the primal
    solution as well as the optimizer output for the dual.
    """

    def optfun(lambd):
        return -maxent_dual(lambd, b, A)

    gradfun = jax.grad(optfun)

    bounds = []
    for _ in range(n_inequality):
        bounds.append((0, None))
    for _ in range(A.shape[0] - n_inequality):
        bounds.append((None, None))

    if not opt_kwargs:
        opt_kwargs = {}

    res = scipy.optimize.minimize(
        optfun,
        jnp.zeros(A.shape[0]),
        jac=gradfun,
        bounds=bounds,
        method="L-BFGS-B",
        **opt_kwargs,
    )

    return {"primal": maxent_solution(res.x, A), "dual_opt_res": res}


class MaxentDualLbfgsSolver(GeneralizedDivergenceSolver):
    def __init__(self, mom_atol=0, mom_rtol=0, solve_kwargs=None):
        self.mom_atol = mom_atol
        self.mom_rtol = mom_rtol

        if not solve_kwargs:
            solve_kwargs = {}
        self.solve_kwargs = solve_kwargs

        self.opt_res_list = []

    def _weights1sample(self, i):
        return self.opt_res_list[i]["primal"]

    def _fit1sample(self, X, mu):
        if self.mom_atol == 0 and self.mom_rtol == 0:
            A = -X.T
            b = -mu
            n_inequality = 0
        else:
            eps = self.mom_atol + self.mom_rtol * jnp.abs(mu)
            A = jnp.vstack([X.T, -X.T])
            b = jnp.concatenate([mu + eps, -mu + eps])
            n_inequality = A.shape[0]

        res = maxent_solve_dual_lbfgs(A, b, n_inequality, opt_kwargs=self.solve_kwargs)

        logger.info(
            f"objective={res['dual_opt_res']['fun']}, {res['dual_opt_res']['message']}"
        )

        return res

    def _objective(self):
        w = self.weights(renormalize=False)
        return jnp.sum(w * jnp.log(w), axis=0) - jnp.log(w.shape[0])

    def _dual_moments(self):
        vals = []
        for i in range(len(self.opt_res_list)):
            vals.append(self.opt_res_list[i]["dual_opt_res"].x)
        vals = jnp.array(vals)

        if self._equality_constraints:
            # Note we use -X, -mu in the equality case
            return {"dual": -vals}
        else:
            nvals = vals.shape[1] / 2
            assert nvals == round(nvals)
            nvals = int(nvals)
            # FIXME Don't make the fact that upper is the first half
            # manually hardcoded?
            upper = vals[:, :nvals]
            lower = vals[:, nvals:]
            return {"dual": upper - lower, "dual_upper": upper, "dual_lower": lower}
