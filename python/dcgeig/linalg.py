#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import numbers

import dcgeig
import dcgeig.utils as utils

import numpy as NP
import numpy.linalg as NL
import numpy.matlib as ML

import scipy.linalg as SL
import scipy.sparse as SS
import scipy.sparse.linalg as LA



class SuperLL:
    def __init__(self, LU):
        self.LU = LU
        self.shape = LU.shape
        self.matvec = LU.solve
        self.solve = LU.solve


def spll(A):
    assert SS.isspmatrix(A)
    assert utils.is_hermitian(A)

    options = {'SymmetricMode': True}
    LU = LA.splu(A, diag_pivot_thresh=0.0, options=options)

    return SuperLL(LU)



def orthogonalize(V, do_overwrite=False):
    assert isinstance(V, ML.matrix)
    assert V.shape[0] >= V.shape[1]

    W = V if do_overwrite else V.copy()

    A = V.H * V
    L = NL.cholesky(A)

    SL.solve_triangular(L, W.H, lower=True, overwrite_b=True)

    return None if do_overwrite else W



def rayleigh_ritz(K, M, S=None):
    assert SS.isspmatrix(K)
    assert SS.isspmatrix(M)

    if S is None:
        A = K.todense()
        B = M.todense()
        Q = SS.identity(K.shape[0], dtype=K.dtype)
    else:
        assert isinstance(S, ML.matrix)
        assert S.shape[0] > S.shape[1]

        Q, _ = SL.qr(S, mode='economic')
        Q = ML.matrix(Q)

        A = utils.force_hermiticity(Q.H * K * Q)
        B = utils.force_hermiticity(Q.H * M * Q)

    d, X_Q = dcgeig.deflation(A, B)

    t = NP.isfinite(d)
    d = d[t]
    X_Q = X_Q[:,t]

    i = NP.argsort(d)
    d = d[i]
    X_Q = X_Q[:,i]

    X = Q * X_Q

    return d, X



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
