
import pytest

import os

import numpy as np
import cvxpy as cp
import quipcell as qpc

import quipcell.maxent_dual_lbfgs as maxent_dual_lbfgs

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
        x, mu, mom_atol=.001,
        solve_kwargs={'solver': cp.OSQP}
    )

    assert np.allclose(w, w2)

def test_example_relaxed2():
    w = np.loadtxt(os.path.join(
        dirname,
        'test_example_w_relaxed2.txt'
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
        x, mu, mom_rtol=.1,
        solve_kwargs={'solver': cp.OSQP}
    )

    assert np.allclose(w, w2)

def test_dual_small_epsilon():
    x = np.loadtxt(os.path.join(
        dirname,
        'test_example_x.txt'
    ))

    mu = np.loadtxt(os.path.join(
        dirname,
        'test_example_mu.txt'
    ))

    res1 = qpc.AlphaDivergenceCvxpySolver(
        alpha=2, mom_atol=1e-8,
        solve_kwargs={'solver': cp.OSQP},
    )
    res1.fit(x, mu)

    res2 = qpc.AlphaDivergenceCvxpySolver(
        alpha=2,
        #mom_atol=.001, mom_rtol=.001,
        solve_kwargs={'solver': cp.OSQP},
    )
    res2.fit(x, mu)

    l2 = res2._dual_moments()
    l1 = res1._dual_moments()

    assert np.allclose(l1['dual'], l2['dual'], rtol=.1, atol=.001)

    assert np.allclose(res1.dual_sum1(), res2.dual_sum1(),
                       rtol=1e-2, atol=1e-4)

    assert np.allclose(res1.dual_nonneg(), res2.dual_nonneg(),
                       rtol=1e-2, atol=5e-4)

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
    #assert np.allclose(w, w2, atol=1e-6, rtol=1e-4)

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
    #assert np.allclose(w, w2, atol=1e-6, rtol=1e-3)

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
        use_dual_lbfgs=False,
        solve_kwargs={'solver': cp.ECOS}
    )

    assert np.allclose(w, w2)

def test_maxent_dual_lbfgs_eq():
    x = np.loadtxt(os.path.join(
        dirname,
        'test_example_x.txt'
    ))

    mu = np.loadtxt(os.path.join(
        dirname,
        'test_example_mu.txt'
    ))

    w = qpc.estimate_weights_multisample(
        x, mu, alpha='kl',
        use_dual_lbfgs=False,
        solve_kwargs={'solver': cp.ECOS},
        return_with_dual=True
    )

    w2 = qpc.estimate_weights_multisample(
        x, mu, alpha='kl',
        #use_dual_lbfgs=True,
        return_with_dual=True
    )

    assert np.allclose(w['weights'], w2['weights'], atol=1e-5)
    assert np.allclose(w['dual'], w2['dual'], atol=1e-2, rtol=1e-2)


def test_maxent_dual_lbfgs_ineq():
    x = np.loadtxt(os.path.join(
        dirname,
        'test_example_x.txt'
    ))

    mu = np.loadtxt(os.path.join(
        dirname,
        'test_example_mu.txt'
    ))

    w = qpc.estimate_weights_multisample(
        x, mu, alpha='kl',
        mom_atol=.01,
        use_dual_lbfgs=False,
        solve_kwargs={'solver': cp.ECOS},
        return_with_dual=True
    )

    w2 = qpc.estimate_weights_multisample(
        x, mu, alpha='kl',
        mom_atol=.01,
        #use_dual_lbfgs=True,
        return_with_dual=True
    )

    assert np.allclose(w['weights'], w2['weights'], atol=1e-5)
    assert np.allclose(w['dual'], w2['dual'], atol=1e-2, rtol=1e-2)
    assert np.allclose(w['dual_lower'], w2['dual_lower'], atol=1e-2, rtol=1e-2)
    assert np.allclose(w['dual_upper'], w2['dual_upper'], atol=1e-2, rtol=1e-2)
