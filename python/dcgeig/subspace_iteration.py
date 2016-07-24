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

import scipy.linalg as SL
import scipy.sparse as SS

import numbers

import time



def chebychev(degree, tau, c, e, solve, K, M, X):
    assert isinstance(tau, numbers.Real)
    assert isinstance(c, numbers.Real)
    assert isinstance(e, numbers.Real)
    assert e > 0
    assert isinstance(degree, int)
    assert degree >= 0

    P0 = X
    P1 = solve( (M-c*K)*X ) / (tau - c)

    a0 = e / (tau-c)
    a1 = a0

    for k in xrange(degree-1):
        a2 = 1 / (2.0/a0 - a1)
        P2 = 2*a2/e * solve( (M-c*K)*P1 ) - a2*a1 * P0

        a1 = a2
        del a2

        P0 = P1
        P1 = P2
        del P2

    return P1



def inverse_iteration( \
        lambda_c, degree, solve, K, M, d, B,
        block_size=256, overwrite_b=False):
    assert isinstance(lambda_c, numbers.Real)
    assert lambda_c > 0
    assert isinstance(degree, int)
    assert NP.isrealobj(d)
    assert d.size == B.shape[1]
    assert isinstance(block_size, int)
    assert block_size > 0
    assert isinstance(overwrite_b, bool)

    m = d.size

    min_d = min(d)
    max_d = max(d)
    median_d = NP.median(d)

    tau = 1/min(min_d, lambda_c)
    z = max(min_d, lambda_c)
    a = max(2*z, median_d)
    b = max(max_d, 2*a)
    c = (1/a + 1/b) / 2
    e = (1/a - 1/b) / 2

    assert a > 0
    assert b > a
    assert c > 0
    assert tau > c+e

    X = B if overwrite_b else ML.copy(B)

    for l in xrange(0, m, block_size):
        r = min(l+block_size, m)

        X[:,l:r] = chebychev(degree, tau, c, e, solve, K, M, X[:,l:r])

    return None if overwrite_b else X



def execute(options, lambda_c, do_stop, LU, K, M, d, X):
    assert isinstance(options, tools.Options)
    assert SS.isspmatrix_csc(K)
    assert utils.is_hermitian(K)
    assert SS.isspmatrix(M)
    assert utils.is_hermitian(M)

    poly_degree = 2

    f = LU.solve

    wallclock_time_sle = 0
    wallclock_time_rr = 0

    for i in range(1, options.max_num_iterations+1):
        wallclock_time_sle_start = time.time()
        inverse_iteration(lambda_c, poly_degree, f, K, M, d, X, overwrite_b=True)
        wallclock_time_sle += time.time() - wallclock_time_sle_start

        wallclock_time_rr_start = time.time()
        d[:], X[:,:], eta, delta = tools.rayleigh_ritz(K, M, X)
        wallclock_time_rr += time.time() - wallclock_time_rr_start

        if do_stop(d, X, eta, delta):
            break

    assert not NP.any(NP.isinf(d))
    assert not NP.any(NP.isnan(d))

    return i, wallclock_time_sle, wallclock_time_rr
