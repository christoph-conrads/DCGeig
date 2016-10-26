#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import dcgeig.utils as utils
import dcgeig.multilevel_tools as tools

import numpy as NP
import numpy.matlib as ML
import numpy.polynomial.chebyshev as NPC

import scipy.linalg as SL
import scipy.sparse as SS

import numbers

import time



def compute_jackson_coefficients(d):
    assert isinstance(d, int)
    assert d >= 0

    if d == 0:
        return NP.full(1, 1.0)

    k = 1.0 * NP.arange(0, d+1)
    r = NP.pi / (d+1)
    cs = (d+1-k) * NP.cos(k*r) + NP.sin(k*r) / NP.tan(r)
    cs[-1] = 0

    assert cs.size == d+1

    return cs / (d+1)



def compute_chebyshev_heaviside_coefficients(d):
    assert isinstance(d, int)
    assert d >= 0

    k = 1.0 * NP.arange(1, d+1)
    cs = NP.full(d+1, NP.nan)
    cs[0] = 0.5
    cs[1::4] = 1.0
    cs[2::4] = 0.0
    cs[3::4] = -1.0
    cs[4::4] = 0.0
    cs[1:] = cs[1:] * 2/(NP.pi * k)

    return NPC.chebtrim(cs)



def evaluate_matrix_polynomial(ps, s, solve, K, M, X):
    assert isinstance(ps, NP.ndarray)

    d = len(ps)-1
    V = ps[d] * X

    for i in range(d-1, -1, -1):
        V = ps[i] * X + solve( (s*M-K)*V )

    return V



def inverse_iteration( \
        lambda_c, degree, solve, K, M, d, B,
        block_size=256, overwrite_b=False):
    assert isinstance(lambda_c, numbers.Real)
    assert lambda_c > 0
    assert isinstance(degree, int)
    assert degree > 0
    assert NP.isrealobj(d)
    assert d.size == B.shape[1]
    assert isinstance(block_size, int)
    assert block_size > 0
    assert isinstance(overwrite_b, bool)

    X = B if overwrite_b else ML.copy(B)

    degree = 7
    js = compute_jackson_coefficients(degree)
    cs = compute_chebyshev_heaviside_coefficients(degree)
    ps = NPC.cheb2poly( NPC.chebtrim(cs * js) )

    m = d.size

    for l in xrange(0, m, block_size):
        r = min(l+block_size, m)

        X[:,l:r] = \
            evaluate_matrix_polynomial(ps, 2*max(d), solve, K, M, X[:,l:r])

    return None if overwrite_b else X



def execute(options, lambda_c, do_stop, LU, K, M, d, X, eta, delta):
    assert isinstance(options, tools.Options)
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

    for i in range(1, options.max_num_iterations+1):
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
