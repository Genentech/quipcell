
import pytest

import os

import numpy as np
import cvxpy as cp
import quipcell as qpc

dirname = os.path.dirname(os.path.realpath(__file__))

def test_example():
    w = np.loadtxt(os.path.join(
        dirname,
        'test_example_w.txt'
    ))

    x = np.loadtxt(os.path.join(
        dirname,
        'test_example_x.txt'
    ))

    mu = np.loadtxt(os.path.join(
        dirname,
        'test_example_mu.txt'
    ))

    w2 = qpc.estimate_weights_multisample(
        x, mu,
        solve_kwargs={'solver': cp.OSQP}
    )

    assert np.allclose(w, w2)

def test_example_relaxed():
    w = np.loadtxt(os.path.join(
        dirname,
        'test_example_w_relaxed.txt'
    ))

    x = np.loadtxt(os.path.join(
        dirname,
        'test_example_x.txt'
    ))

    mu = np.loadtxt(os.path.join(
        dirname,
        'test_example_mu.txt'
    ))

    w2 = qpc.estimate_weights_multisample(
        x, mu, relax_moment_condition=.001,
        solve_kwargs={'solver': cp.OSQP}
    )

    assert np.allclose(w, w2)

def test_example_norm():
    w = np.loadtxt(os.path.join(
        dirname,
        'test_example_w_norm.txt'
    ))

    x = np.loadtxt(os.path.join(
        dirname,
        'test_example_x.txt'
    ))

    mu = np.loadtxt(os.path.join(
        dirname,
        'test_example_mu.txt'
    ))

    w2 = qpc.estimate_weights_multisample(
        x, mu, use_norm=True,
        solve_kwargs={'solver': cp.ECOS}
    )

    assert np.allclose(w, w2)

def test_example_alpha3():
    w = np.loadtxt(os.path.join(
        dirname,
        'test_example_w_alpha3.txt'
    ))

    x = np.loadtxt(os.path.join(
        dirname,
        'test_example_x.txt'
    ))

    mu = np.loadtxt(os.path.join(
        dirname,
        'test_example_mu.txt'
    ))

    w2 = qpc.estimate_weights_multisample(
        x, mu, alpha=3,
        solve_kwargs={'solver': cp.ECOS}
    )

    assert np.allclose(w, w2)

def test_example_kl():
    w = np.loadtxt(os.path.join(
        dirname,
        'test_example_w_kl.txt'
    ))

    x = np.loadtxt(os.path.join(
        dirname,
        'test_example_x.txt'
    ))

    mu = np.loadtxt(os.path.join(
        dirname,
        'test_example_mu.txt'
    ))

    w2 = qpc.estimate_weights_multisample(
        x, mu, alpha='kl',
        solve_kwargs={'solver': cp.ECOS}
    )

    assert np.allclose(w, w2)
