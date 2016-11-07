#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import numbers

import dcgeig.polynomial as polynomial
import dcgeig.utils as utils

import numpy as NP

import scipy.linalg as SL
import scipy.sparse as SS

import time



def inverse_iteration( \
        solve, K, M, B, sigma,
        polynomial_degree=7, block_size=256, overwrite_b=False):
    assert callable(solve)
    assert SS.isspmatrix(K)
    assert SS.isspmatrix(M)
    assert isinstance(B, NP.ndarray)
    assert isinstance(sigma, numbers.Real)
    assert sigma > 0
    assert isinstance(polynomial_degree, int)
    assert polynomial_degree > 0
    assert isinstance(block_size, int)
    assert block_size > 0
    assert isinstance(overwrite_b, bool)

    X = B if overwrite_b else NP.copy(B)

    a = 0
    b = 1/sigma
    c = (b + a) / 2
    e = (b - a) / 2

    assert c > 0
    assert e > 0

    cs = polynomial.compute_chebyshev_heaviside_coefficients(polynomial_degree)
    js = polynomial.compute_jackson_coefficients(polynomial_degree)
    ps = js * cs

    c1 = 2 * sigma
    c2 = 1
    eval_poly = polynomial.evaluate_matrix_polynomial

    m = B.shape[1]

    for l in xrange(0, m, block_size):
        r = min(l+block_size, m)

        X[:,l:r] = eval_poly(ps, c1, c2, solve, K, M, X[:,l:r])

    return None if overwrite_b else X



def execute(max_num_iterations, lambda_c, do_stop, LU, K, M, d, X, eta, delta):
    assert isinstance(max_num_iterations, int)
    assert max_num_iterations > 0
    assert SS.isspmatrix_csc(K)
    assert utils.is_hermitian(K)
    assert SS.isspmatrix(M)
    assert utils.is_hermitian(M)
    assert d.size == X.shape[1]
    assert d.size == eta.size
    assert d.size == delta.size

    poly_degree = 2

    f = LU.solve

    wallclock_time_sle = 0
    wallclock_time_rr = 0

    for i in range(1, max_num_iterations+1):
        wallclock_time_sle_start = time.time()
        inverse_iteration( \
            lambda_c, poly_degree, f, K, M, d, X, overwrite_b=True)
        wallclock_time_sle += time.time() - wallclock_time_sle_start

        wallclock_time_rr_start = time.time()
        d[:], X[:,:], eta[:], delta[:] = tools.rayleigh_ritz(K, M, X)
        wallclock_time_rr += time.time() - wallclock_time_rr_start

        if do_stop(d, X, eta, delta):
            break

    assert not NP.any(NP.isinf(d))
    assert not NP.any(NP.isnan(d))

    return i, wallclock_time_sle, wallclock_time_rr
