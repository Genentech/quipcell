
import scipy
import scipy.optimize

import jax
import jax.numpy as jnp

def maxent_dual(lambd, b, A):
    return - b @ lambd - jnp.log(jnp.sum(jnp.exp(- A.T @ lambd)))

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
        optfun, jnp.zeros(A.shape[0]),
        jac=gradfun, bounds=bounds,
        method='L-BFGS-B',
        **opt_kwargs
    )

    return {
        "primal": maxent_solution(res.x, A),
        "dual_opt_res": res
    }

def estimate_weights_maxent_dual_lbfgs(
        X, mu,
        mom_atol=0, mom_rtol=0,
        opt_kwargs=None
):
    if mom_atol == 0 and mom_rtol == 0:
        A = -X.T
        b = -mu
        n_inequality = 0
    else:
        eps = mom_atol + mom_rtol * jnp.abs(mu)
        A = jnp.vstack([X.T, -X.T])
        b = jnp.concatenate([mu+eps, -mu+eps])
        n_inequality = A.shape[0]

    return maxent_solve_dual_lbfgs(A, b, n_inequality,
                                   opt_kwargs=opt_kwargs)
        
