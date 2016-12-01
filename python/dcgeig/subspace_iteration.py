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



def heaviside_chebyshev( \
        solve, K, M, B, omega,
        polynomial_degree, block_size=256, overwrite_b=False):
    assert callable(solve)
    assert SS.isspmatrix(K)
    assert SS.isspmatrix(M)
    assert isinstance(B, NP.ndarray)
    assert K.shape[0] == B.shape[0]
    assert M.shape[0] == B.shape[0]
    assert isinstance(omega, numbers.Real)
    assert omega > 0
    assert isinstance(polynomial_degree, int)
    assert polynomial_degree > 0
    assert isinstance(block_size, int)
    assert block_size > 0
    assert isinstance(overwrite_b, bool)

    X = B if overwrite_b else NP.copy(B)

    a = 0
    b = 1/omega
    c = (b + a) / 2
    e = (b - a) / 2

    assert c > 0
    assert e > 0

    cs = polynomial.compute_chebyshev_heaviside_coefficients(polynomial_degree)
    js = polynomial.compute_jackson_coefficients(polynomial_degree)
    ps = js * cs

    c1 = 2 * omega
    c2 = 1
    eval_poly = polynomial.evaluate_matrix_polynomial

    m = B.shape[1]

    for l in xrange(0, m, block_size):
        r = min(l+block_size, m)

        X[:,l:r] = eval_poly(ps, c1, c2, solve, K, M, X[:,l:r])

    return None if overwrite_b else X



def minmax_chebyshev( \
        solve, K, M, B, omega,
        polynomial_degree, block_size=256, overwrite_b=False):
    assert callable(solve)
    assert SS.isspmatrix(K)
    assert SS.isspmatrix(M)
    assert isinstance(B, NP.ndarray)
    assert K.shape[0] == B.shape[0]
    assert M.shape[0] == B.shape[0]
    assert isinstance(omega, numbers.Real)
    assert omega > 0
    assert isinstance(polynomial_degree, int)
    assert polynomial_degree > 0
    assert isinstance(block_size, int)
    assert block_size > 0
    assert isinstance(overwrite_b, bool)

    X = B if overwrite_b else NP.copy(B)

    a = 0
    b = 1/omega
    c = (b + a) / 2
    e = (b - a) / 2

    assert c > 0
    assert e > 0

    d = polynomial_degree
    k = NP.arange(d)
    roots = c + e * NP.cos( NP.pi * (2*k+1) / (2*d) )

    eval_poly = polynomial.evaluate_matrix_polynomial

    m = B.shape[1]

    for l in xrange(0, m, block_size):
        r = min(l+block_size, m)

        Y = X[:,l:r]
        for sigma in roots:
            C = M - sigma * K
            Y = solve(C * Y)
        X[:,l:r] = Y

    return None if overwrite_b else X



def execute( \
        K, M, d, X, lambda_c, eta_max, delta_max, max_num_iterations=10):
    assert SS.isspmatrix(K)
    assert SS.isspmatrix(M)
    assert isinstance(d, NP.ndarray)
    assert NP.all(d > 0)
    assert isinstance(X, NP.ndarray)
    assert X.shape[0] == K.shape[0]
    assert X.shape[1] > 0
    assert X.shape[1] == d.size
    assert isinstance(lambda_c, numbers.Real)
    assert lambda_c > 0
    assert isinstance(eta_max, numbers.Real)
    assert eta_max > 0
    assert eta_max < 1
    assert isinstance(delta_max, numbers.Real)
    assert delta_max > 0
    assert isinstance(max_num_iterations, int)
    assert max_num_iterations > 0

    print 'SI {:4d} {:8.2e}'.format(NP.sum(d <= lambda_c), max(d)/lambda_c)

    LL = linalg.spll(K)
    X = LL.solve(M * X)
    del LL

    for i in range(1, max_num_iterations+1):
        d_old = d
        d, X = linalg.rayleigh_ritz(K, M, X)
        eta, delta = error_analysis.compute_errors(K, M, d, X)


        t = d <= lambda_c
        u = d - delta <= lambda_c
        n_c = NP.sum(t)
        t[:n_c+10] = True
        t = t & u

        if not NP.any(t):
            break


        n = K.shape[0]
        print 'SI {:d} {:2d}  {:4d}'.format(n, i, NP.sum(d <= lambda_c))
        fmt = 'SI {:d} {:2d}  {:8.2e} {:8.2e} {:8.2e} {:8.2e} {:8.2e}'

        be = NP.sort(eta)
        print fmt.format( \
            K.shape[0], i,
            be[0], NP.percentile(be, 25), NP.median(be),
            NP.percentile(be,75), be[-1])

        fe = NP.sort(delta/d)
        print fmt.format( \
            K.shape[0], i,
            fe[0], NP.percentile(fe, 25), NP.median(fe),
            NP.percentile(fe,75), fe[-1])

        nominator = NP.abs(d - d_old)
        denominator = NP.minimum(d_old, d)
        rd = NP.sort(nominator/denominator)

        print fmt.format( \
            K.shape[0], i,
            rd[0], NP.percentile(rd, 25), NP.median(rd),
            NP.percentile(rd,75), rd[-1])

        e = NP.sort(d/lambda_c)
        print fmt.format( \
            K.shape[0], i,
            e[0], NP.percentile(e,25), NP.median(e), NP.percentile(e,75), e[-1])


        if max(eta[t]) < eta_max and max(delta[t]/d[t]) < delta_max:
            break


        t1 = ((eta >= eta_max) | (delta >= d*delta_max)) & (d <= lambda_c)
        t2 = d > lambda_c
        t0 = ~(t1 | t2)

        n_s = d.size
        xs = NP.arange(n_s)
        p = NP.concatenate( [xs[t0], xs[t1], xs[t2]] )

        d = d[p]
        X = X[:,p]
        del eta
        del delta

        l = NP.sum(t0)
        r = NP.sum(t0) + NP.sum(t1)
        assert r + NP.sum(t2) == d.size
        assert r == NP.sum(d <= lambda_c)

        print 'SI {:4d} l={:d} r={:d}'.format(n_s, l, r)

        LU = LA.splu(K - 0.9*lambda_c * M, options={'SymmetricMode': True})
        X[:,r:] = LU.solve(M * X[:,r:])
        X[:,r:] = LU.solve(M * X[:,r:])
        del LU

        LL = linalg.spll(K)
        X[:,l:] = LL.solve(M * X[:,l:])
        X[:,l:] = LL.solve(M * X[:,l:])



    assert not NP.any(NP.isinf(d))
    assert not NP.any(NP.isnan(d))

    return d[:n_c], X[:,:n_c], eta[:n_c], delta[:n_c]
