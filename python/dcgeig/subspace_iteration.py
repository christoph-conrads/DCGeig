#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import dcgeig.utils as utils

import numpy as NP
import numpy.matlib as ML

import scipy.linalg as SL
import scipy.sparse as SS

import numbers

import time



def chebychev(degree, c, e, solve, K, M, X):
    assert isinstance(degree, int)
    assert degree >= 0
    assert isinstance(c, numbers.Real)
    assert isinstance(e, numbers.Real)
    assert e > 0

    ks = NP.arange(degree)
    roots = c + e * NP.cos( (2*ks+1)/(2.0*degree) * NP.pi )

    for k in ks:
        X = solve( (M-roots[k]*K)*X )

    return X



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

    # compute ellipse
    a = max(d)
    c = (1/a + 0) / 2
    e = (1/a - 0) / 2

    assert a > 0
    assert c > 0
    assert e > 0

    m = d.size

    for l in xrange(0, m, block_size):
        r = min(l+block_size, m)

        X[:,l:r] = chebychev(degree, c, e, solve, K, M, X[:,l:r])

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
