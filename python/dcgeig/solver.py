#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import numbers

import copy

import numpy as NP
import numpy.linalg as NL
import numpy.matlib as ML
import numpy.random

import scipy.sparse as SS
import scipy.sparse.linalg as LA

import dcgeig.linalg as linalg
import dcgeig.polynomial as polynomial
import dcgeig.utils as utils



def estimate_trace(f, n, b, dtype=NP.float64):
    assert callable(f)
    assert isinstance(n, int)
    assert n > 0
    assert isinstance(b, int)
    assert b > 0
    assert b <= n


    random = numpy.random.RandomState(seed=1)

    V = 2 * random.randint(0, 2, [n,b]).astype(dtype) - 1
    W = f(V)
    xs = NP.einsum('ij,ij->j', V, W, casting='no')

    mean = NP.mean(xs)
    std = NP.std(xs, ddof=1)

    return mean, std



def estimate_eigenvalue_count(K, M, lambda_1, lambda_c, d, b):
    assert SS.isspmatrix(K)
    assert SS.isspmatrix(M)
    assert isinstance(lambda_1, numbers.Real)
    assert lambda_1 > 0
    assert isinstance(lambda_c, numbers.Real)
    assert lambda_c > lambda_1
    assert isinstance(d, int)
    assert d > 0
    assert isinstance(b, int)
    assert b > 0

    n = K.shape[0]
    f = polynomial.approximate_projection(d, lambda_1, lambda_c, K, M)

    mean, std = estimate_trace(f, n, b, K.dtype)

    return mean, std



def compute_largest_eigenvalue(K, M, S, tol=0):
    assert SS.isspmatrix(K)
    assert utils.is_hermitian(K)
    assert SS.isspmatrix(M)
    assert utils.is_hermitian(M)
    assert isinstance(S, ML.matrix)
    assert K.shape[0] == S.shape[0]
    assert K.shape[0] > S.shape[1]
    assert isinstance(tol, numbers.Real)
    assert tol >= 0
    assert tol < 1

    m = S.shape[1]


    if m == 1:
        a = (S.H * K * S)[0,0]
        b = (S.H * M * S)[0,0]
        return a/b


    B = S.H * (M * S)
    L = NL.cholesky(B)

    def f(self, u):
        assert u.shape[0] == m

        v = NL.solve(L.H, u)
        v = NP.reshape(v, [m,1])
        v = S.H * (K * (S * v))
        v = NL.solve(L, v)

        return v

    Solver = type('InlineGEPSolver', (object,), {'matvec': f, 'shape': (m,m)})
    operator = LA.aslinearoperator( Solver() )

    v0 = NP.ones([m,1], dtype=K.dtype)

    d = LA.eigsh(operator, k=1, v0=v0, tol=tol, return_eigenvectors=False)

    return max(d)
