#!/usr/bin/env python

import cvxpy as cp
import numpy as np

import quipcell as qpc

mu1 = np.array([1, 0])
mu2 = np.array([0, 1])
sigma = 0.1

mu = np.array([[0.75, 0.25], [0.5, 0.5], [0.1, 0.9]])

assert np.allclose(mu.sum(axis=1), 1)

rng = np.random.default_rng(12345)

n = 10

x1 = mu1 + sigma * rng.normal(size=(n, 2))
x2 = mu2 + sigma * rng.normal(size=(n, 2))

x = np.vstack([x1, x2])

# Explicitly specify solver b/c cvxpy may change default solvers,
# e.g. in 1.5 clarabel will become default for many problems

w = qpc.estimate_weights_multisample(x, mu, solve_kwargs={"solver": cp.OSQP})

w_relax = qpc.estimate_weights_multisample(
    x, mu, mom_atol=0.001, solve_kwargs={"solver": cp.OSQP}
)

w_relax2 = qpc.estimate_weights_multisample(
    x, mu, mom_rtol=0.1, solve_kwargs={"solver": cp.OSQP}
)

w_norm = qpc.estimate_weights_multisample(
    x, mu, use_norm=True, solve_kwargs={"solver": cp.ECOS}
)

w3 = qpc.estimate_weights_multisample(x, mu, alpha=3, solve_kwargs={"solver": cp.ECOS})

w_kl = qpc.estimate_weights_multisample(
    x, mu, alpha="kl", solve_kwargs={"solver": cp.ECOS}
)


def check_weights_reasonable(w):
    assert np.allclose(w.sum(axis=0), 1)

    assert np.all(n * w[:n, 0] < 0.85) and np.all(n * w[:n, 0] > 0.65)
    assert np.all(n * w[n:, 0] < 0.35) and np.all(n * w[n:, 0] > 0.15)

    assert np.all(n * w[:, 1] > 0.4) and np.all(n * w[:, 1] < 0.6)

    assert np.all(n * w[:n, 2] < 0.25) and np.all(n * w[n:, 2] > 0.8)


check_weights_reasonable(w)
check_weights_reasonable(w_relax)
check_weights_reasonable(w_norm)

# check_weights_reasonable(w_relax2)
assert np.max(np.abs(w_relax2 - w)) < 0.01

# check_weights_reasonable(w3)
assert np.max(np.abs(w3 - w)) < 0.01

# check_weights_reasonable(w_kl)
assert np.max(np.abs(w_kl - w)) < 0.012

np.savetxt("test_example_mu.txt", mu)
np.savetxt("test_example_x.txt", x)
np.savetxt("test_example_w.txt", w)
np.savetxt("test_example_w_relaxed.txt", w_relax)
np.savetxt("test_example_w_relaxed2.txt", w_relax2)
np.savetxt("test_example_w_norm.txt", w_norm)
np.savetxt("test_example_w_alpha3.txt", w3)
np.savetxt("test_example_w_kl.txt", w_kl)
