#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import numbers

import dcgeig.error_analysis as error_analysis
import dcgeig.linalg as linalg
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
    assert K.shape[0] == B.shape[0]
    assert M.shape[0] == B.shape[0]
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



def execute(solve, K, M, X, lambda_c, eta_max, delta_max,max_num_iterations=10):
    assert callable(solve)
    assert SS.isspmatrix(K)
    assert SS.isspmatrix(M)
    assert isinstance(X, NP.ndarray)
    assert X.shape[0] == K.shape[0]
    assert isinstance(lambda_c, numbers.Real)
    assert lambda_c > 0
    assert isinstance(eta_max, numbers.Real)
    assert eta_max > 0
    assert eta_max < 1
    assert isinstance(delta_max, numbers.Real)
    assert delta_max > 0
    assert delta_max < 1
    assert isinstance(max_num_iterations, int)
    assert max_num_iterations > 0

    d_max = linalg.compute_largest_eigenvalue(K, M, X)

    for i in range(1, max_num_iterations+1):
        inverse_iteration(solve, K, M, X, 2*d_max, overwrite_b=True)

        d, X = linalg.rayleigh_ritz(K, M, X)
        eta, delta = error_analysis.compute_errors(K, M, d, X)

        t = d-delta <= lambda_c

        if max(eta[t]) < eta_max and max(delta[t]/d[t]) < delta_max:
            break

        d_max = max(d)

    assert not NP.any(NP.isinf(d))
    assert not NP.any(NP.isnan(d))

    return d[t], X[:,t]
