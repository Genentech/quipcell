import logging
import numpy as np
import cvxpy as cp

logger = logging.getLogger(__name__)

class GeneralizedDivergenceSolver(object):
    def fit(self, X, mu_multisample):
        """TODO docstring"""
        self.opt_res_list = []
        n = mu_multisample.shape[0]
        for i in range(n):
            logger.info(f"Solving for sample i={i}")

            opt_res = self._fit1sample(
                X, mu_multisample[i,:]
            )

            self.opt_res_list.append(opt_res)
        
    # TODO: Add option to incorporate size_factors?
    def weights(self, renormalize=True):
        if not self.opt_res_list:
            raise ValueError("Need to call fit() first")

        w_hat_multisample = []
        for i in range(len(self.opt_res_list)):
            w_hat_multisample.append(self._weights1sample(i))
            
        ret = np.array(w_hat_multisample).T
        if renormalize:
            ret[ret < 0] = 0
            ret = np.einsum("ij,j->ij", ret, 1.0 / ret.sum(axis=0))

        return ret
    

class AlphaDivergenceCvxpySolver(GeneralizedDivergenceSolver):
    def __init__(self, alpha,
                 mom_atol=0, mom_rtol=0,
                 solve_kwargs=None,
                 use_norm=False):
        """
        :param float alpha: Value of alpha for alpha-divergence. Also accepts 'pearson' for alpha=2 (which is a quadratic program) or 'kl' for alpha=1 (which is same as maximum entropy).
        :param float mom_atol: For moment constraints, require X.T @ w = mu +/- eps, where eps = mom_atol + abs(mu) * mom_rtol.
        :param float mom_rtol: For moment constraints, require X.T @ w = mu +/- eps, where eps = mom_atol + abs(mu) * mom_rtol.
        :param dict solve_kwargs: Additional kwargs to pass to `cvxpy.Problem.solve`.
        :param bool use_norm: Whether to optimize the pnorm sum(w**alpha)**(1/alpha) instead of sum(w**alpha). While mathematically equivalent when alpha > 1, the conditioning of the optimization problem may be better with pnorm objective. However, it prevents using efficient quadratic optimization solvers when alpha=2. See here for discussion: http://cvxr.com/cvx/doc/advanced.html#eliminating-quadratic-forms
        """
        if alpha == 'pearson':
            alpha = 2
        elif alpha == 'kl':
            alpha = 1
        elif type(alpha) == str:
            raise ValueError(f'Unrecognized divergence {alpha}')

        self.alpha = alpha
        self.mom_atol = mom_atol
        self.mom_rtol = mom_rtol

        # Reference for solver options:
        # https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options
        if not solve_kwargs:
            solve_kwargs = {}
        solve_kwargs.setdefault("verbose", False)
        self.solve_kwargs = solve_kwargs

        if use_norm:
            if alpha <= 1:
                raise NotImplementedError('use_norm requires alpha > 1')
        self.use_norm = use_norm
        
        self.opt_res_list = []

    def _weights1sample(self, i):
        """TODO docstring"""
        w_hat, = self.opt_res_list[i].variables()
        return w_hat.value.copy()

    # TODO: Add accelerated projected gradient descent solver? E.g. see:
    # https://stackoverflow.com/questions/65526377/cvxpy-returns-infeasible-inaccurate-on-quadratic-programming-optimization-proble
    def _fit1sample(self, X, mu):
        """Estimate density weights for a single sample on a single-cell reference.

        :param `numpy.ndarray` X: Reference embedding. Rows=cells, columns=features.
        :param `numpy.ndarray` mu: Sample moments. Either bulk gene counts (for bulk deconvolution) or sample centroids of single cells (for differential abundance). Should be a 1-dimensional array.

        :rtype: :class:`cvxpy.Problem`
        """

        n = X.shape[0]
        z = np.zeros(n)

        w = cp.Variable(n)

        # Initialize as uniform distribution
        # NOTE: Unclear in which solvers CVXPY will actually use the initialization
        # https://www.cvxpy.org/tutorial/advanced/index.html#warm-start
        # https://stackoverflow.com/questions/52314581/initial-guess-warm-start-in-cvxpy-give-a-hint-of-the-solution
        w.value = np.ones(n) / n

        Xt = X.T

        # TODO Scale the objective to match traditional definition of
        # alpha divergence? I.e. scale it by 1/alpha(alpha-1) and
        # subtract a constant. Even tho the optimization problem is
        # equivalent, the result may be more interpretable using the
        # traditional scaling.
        if self.use_norm:
            objective = cp.Minimize(cp.norm(w, self.alpha))
        elif self.alpha == 1:
            objective = cp.Minimize(-cp.sum(cp.entr(w)))
        elif self.alpha == 0:
            objective = cp.Minimize(-cp.sum(cp.log(w)))
        elif self.alpha == 2:
            objective = cp.Minimize(cp.sum_squares(w))
        elif self.alpha < 1 and self.alpha > 0:
            # sign of alpha*(alpha-1) is negative in this case
            objective = cp.Minimize(-cp.sum(w**self.alpha))
        else:
            objective = cp.Minimize(cp.sum(w**self.alpha))

        constraints = [w >= z, cp.sum(w) == 1.0]
        if self.mom_atol == 0 and self.mom_rtol == 0:
            constraints.append(
                Xt @ w == mu,
            )
        else:
            eps = self.mom_atol + self.mom_rtol * np.abs(mu)
            constraints.append(
                Xt @ w - mu <= eps
            )

            constraints.append(
                Xt @ w - mu >= -eps
            )

        prob = cp.Problem(
            objective,
            constraints
        )

        res = prob.solve(**self.solve_kwargs)
        assert prob.variables()[0] is w
        assert prob.value is res

        # TODO: Reenable assertions, with params for atol/rtol?
        #w_hat, = prob.variables()
        #w_hat = w_hat.value
        #assert np.all(np.isclose(w_hat, 0, atol=1e-4) | (w_hat >= 0))
        #assert np.allclose(np.sum(w_hat), 1)

        logger.info(f"objective={prob.value}, {prob.status}")

        return prob
