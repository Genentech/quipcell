import numpy as np
import cvxpy as cp

# TODO: use logging instead
from datetime import datetime

# TODO: parametrize the cvxpy.Problem by mu to improve efficiency
# (e.g. by reducing compilation times)
# https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming

def estimate_weights_multisample(X, mu_multisample):
    w_hat_multisample = []

    for i in range(mu_multisample.shape[0]):
        print(datetime.now().strftime(f'%H:%M:%S: {i}'))

        prob = estimate_weights(X, mu_multisample[i,:],
                                verbose=False)

        w_hat, = prob.variables()
        w_hat = w_hat.value.copy()

        #norm = prob.value #float
        #status = prob.status #string

        w_hat_multisample.append(w_hat)

    print(datetime.now().strftime('%H:%M:%S: Finished'))
    return np.array(w_hat_multisample).T

def estimate_weights(X, mu, **solve_kwargs):
    n = X.shape[0]
    z = np.zeros(n)

    w = cp.Variable(n)
    Xt = X.T

    prob = cp.Problem(
        cp.Minimize(cp.norm(w, 2)),
        [w >= z,
         Xt @ w == mu,
         cp.sum(w) == 1.0]
    )

    res = prob.solve(**solve_kwargs)
    assert prob.variables()[0] is w
    assert prob.value is res

    return prob
