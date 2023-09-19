#!/usr/bin/env python

import numpy as np
import scDensQP as scdqp

mu1 = np.array([1,0])
mu2 = np.array([0,1])
sigma = .1

mu = np.array([
    [.75, .25],
    [.5, .5],
    [.1, .9]
])

assert np.allclose(mu.sum(axis=1), 1)

rng = np.random.default_rng(12345)

n = 10

x1 = mu1 + sigma * rng.normal(size=(n, 2))
x2 = mu2 + sigma * rng.normal(size=(n, 2))

x = np.vstack([x1, x2])

w = scdqp.estimate_weights_multisample(x, mu)

assert np.allclose(w.sum(axis=0), 1)

assert np.all(n*w[:n, 0] < .85) and np.all(n*w[:n, 0] > .65)
assert np.all(n*w[n:, 0] < .35) and np.all(n*w[n:, 0] > .15)

assert np.all(n*w[:,1] > .4) and np.all(n*w[:,1] < .6)

assert np.all(n*w[:n, 2] < .25) and np.all(n*w[n:, 2] > .8)

np.savetxt("test_example_mu.txt", mu)
np.savetxt("test_example_x.txt", x)
np.savetxt("test_example_w.txt", w)
