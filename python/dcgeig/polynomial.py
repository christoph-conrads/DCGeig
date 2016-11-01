#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import numpy as NP
import numpy.random
import numpy.matlib as ML
import numpy.polynomial.chebyshev as NPC

import scipy.sparse as SS
import scipy.sparse.linalg as LA

import numbers



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

    return cs



def compute_chebyshev_step_coefficients(d, a, b):
    assert isinstance(d, int)
    assert d >= 0
    assert isinstance(a, numbers.Real)
    assert a >= -1
    assert a < +1
    assert isinstance(b, numbers.Real)
    assert b > -1
    assert b <= +1
    assert a < b


    # check for the heaviside function
    if a == 0 and b == 1:
        return compute_chebyshev_heaviside_coefficients(d)


    u = NP.arccos(a)
    v = NP.arccos(b)

    cs = NP.full(d+1, NP.nan)
    k = 1.0 * NP.arange(1, d+1)

    cs[0] = u - v
    cs[1:] = 2 * (NP.sin(k*u) - NP.sin(k*v)) / k

    return cs / NP.pi



# this function implements the Clenshaw algorithm for Chebyshev polynomials
def evaluate_matrix_polynomial(cs, c1, c2, solve, A, B, U):
    assert isinstance(cs, NP.ndarray)
    assert isinstance(c1, numbers.Real)
    assert NP.isfinite(c1)
    assert isinstance(c2, numbers.Real)
    assert NP.isfinite(c2)
    assert callable(solve)
    assert SS.isspmatrix(A)
    assert SS.isspmatrix(B)
    assert isinstance(U, NP.ndarray)

    d = len(cs)-1
    C = c1*B - c2*A

    if d == 0:
        return cs[0] * U
    if d == 1:
        return cs[0] * U + cs[1] * solve(C*U)

    W = cs[-1] * U
    V = cs[-2] * U + 2 * solve(C*W)

    for i in range(d-2, 0, -1):
        T = cs[i] * U + 2 * solve(C*V) - W
        W = V
        V = T

    # note the missing factor 2 in the second term
    return cs[0] * U + solve(C*V) - W



def approximate_projection(d, lambda_1, lambda_c, K, M):
    assert isinstance(d, int)
    assert d > 0
    assert isinstance(lambda_1, numbers.Real)
    assert lambda_1 > 0
    assert isinstance(lambda_c, numbers.Real)
    assert lambda_c > lambda_1
    assert SS.isspmatrix(K)
    assert SS.isspmatrix(M)
    assert NP.all( K.shape == M.shape )


    if 2*lambda_1 != lambda_c:
        raise NotImplementedError('2*lambda_1 != lambda_c')


    n = K.shape[0]
    LU = LA.splu(K, diag_pivot_thresh=0)

    cs = compute_chebyshev_heaviside_coefficients(d)
    js = compute_jackson_coefficients(d)
    ps = NPC.chebtrim(cs * js)

    c1 = 2*lambda_1
    c2 = 1.0

    def f(V):
        return evaluate_matrix_polynomial(ps, c1, c2, LU.solve, K, M, V)

    return f
