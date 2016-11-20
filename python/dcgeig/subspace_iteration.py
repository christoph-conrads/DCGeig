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
import scipy.sparse.linalg as LA

import time

import matplotlib.pyplot as PP



def inverse_iteration( \
        solve, K, M, B, sigma,
        polynomial_degree, block_size=256, overwrite_b=False):
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



def execute( \
        K, M, X, lambda_c, lambda_s, eta_max, delta_max, max_num_iterations=10):
    assert SS.isspmatrix(K)
    assert SS.isspmatrix(M)
    assert isinstance(X, NP.ndarray)
    assert X.shape[0] == K.shape[0]
    assert X.shape[1] > 0
    assert isinstance(lambda_c, numbers.Real)
    assert lambda_c > 0
    assert isinstance(lambda_s, numbers.Real)
    assert lambda_s > lambda_c
    assert isinstance(eta_max, numbers.Real)
    assert eta_max > 0
    assert eta_max < 1
    assert isinstance(delta_max, numbers.Real)
    assert delta_max > 0
    assert isinstance(max_num_iterations, int)
    assert max_num_iterations > 0

    cs = lambda_s

    for i in range(1, max_num_iterations+1):
        t0 = time.time()
        c0 = time.clock()

        options = {'SymmetricMode': True}
        LL0 = LA.splu(SS.csc_matrix(K - lambda_c * M), options=options)
        for k in range(3):
            X = LL0.solve(M * X)
        del LL0

        A = SS.csc_matrix(K + lambda_c * M)
        LL1 = linalg.spll(A)
        inverse_iteration(LL1.solve, A, M, X, cs, 11, overwrite_b=True)
        del LL1
        del A

        c1 = time.clock()
        t1 = time.time()

        d, X = linalg.rayleigh_ritz(K, M, X)
        eta, delta = error_analysis.compute_errors(K, M, d, X)
        c2 = time.clock()
        t2 = time.time()

        fmt = 'SI  {:6d} {:4d}  {:6.1f} {:6.1f}  {:6.1f} {:6.1f}'
        n = X.shape[0]
        m = X.shape[1]
        print fmt.format(n, m, t1-t0, c1-c0, t2-t1, c2-c1)

        PP.figure()
        PP.plot( d/lambda_c, eta )
        PP.plot( d/lambda_c, delta/d )
        axes = PP.gca()
        axes.set_xlim([0, 10])
        PP.yscale('log')
        PP.title( 'Iteration {:d}: Errors over eigenvalues'.format(i) )

        PP.figure()
        PP.plot( NP.arange(m), eta )
        PP.plot( NP.arange(m), delta/d )
        PP.yscale('log')
        PP.title( 'Iteration {:d}: Errors over indices'.format(i) )

        PP.figure()
        PP.plot( NP.arange(m), (d-delta)/lambda_c )
        PP.yscale('log')
        PP.title( 'Iteration {:d}: Eigenvalues'.format(i) )

        t = d <= lambda_c
        u = d - delta <= lambda_c
        n_c = NP.sum(t)
        t[:n_c+10] = True
        t = t & u

        if not NP.any(t):
            break

        if max(eta[t]) < eta_max and max(delta[t]/d[t]) < delta_max:
            break

    PP.show()

    assert not NP.any(NP.isinf(d))
    assert not NP.any(NP.isnan(d))

    return d[:n_c], X[:,:n_c], eta[:n_c], delta[:n_c]
