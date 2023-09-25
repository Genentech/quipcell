
import pytest

import os

import numpy as np
import quickbeam as qb

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

    w2 = qb.estimate_weights_multisample(x, mu)

    assert np.allclose(w, w2)
