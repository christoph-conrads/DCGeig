#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2015-2016 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

import numpy as NP
import numpy.linalg as LA

import scipy.sparse as SS
import scipy.sparse.linalg as SL

import utils



norms = lambda A: LA.norm(A, axis=0)
frobenius_norm = \
    lambda A: SL.norm(A, 'fro') if SS.isspmatrix(A) else LA.norm(A, 'fro')



# columnwise_dot_product(A, B) in Python == dot(A, B) in Matlab
def columnwise_dot_product(A, B):
    assert A.shape == B.shape

    return LA.norm(NP.multiply(NP.conj(A), B), ord=1, axis=0)



# computes sqrt( ||K||_F^2 + |d|^2 ||M||_F^2 )
def weighted_norm_average(k_F, m_F, d):
    assert k_F >= 0
    assert m_F >= 0

    return norms(NP.vstack([NP.full_like(d, k_F), abs(d)*m_F]))



def compute_backward_error(K, M, d, X):
    assert K.dtype == M.dtype
    assert K.dtype == d.dtype
    assert K.dtype == X.dtype
    assert X.shape[1] == d.size

    if not utils.is_hermitian(K) or not utils.is_hermitian(M):
        raise ValueError('Matrices must be Hermitian')

    if NP.any( NP.iscomplex(d) ):
        raise ValueError('Eigenvalues must be real')


    nan = NP.nan
    eps = NP.finfo(K.dtype).eps
    n, m = X.shape

    k_F = frobenius_norm(K)
    m_F = frobenius_norm(M)

    if k_F == 0 and m_F == 0:
        return NP.zeros(m)


    # normalize vectors
    ns = norms(X)
    KX = K*X / ns
    MX = M*X / ns

    eta = NP.full(m, nan)

    # eta for d == -1
    s = d==-1
    norms_KX = norms(KX[:,s])
    rK = norms_KX / k_F
    rK[norms_KX==0] = 0

    norms_MX = norms(MX[:,s])
    rM = norms_MX / m_F
    rM[norms_MX==0] = 0

    eta[s] = norms( NP.vstack([rK, rM]) )


    # eta for d != -1
    R = NP.full_like(X, nan)
    numerator = NP.full(m, nan)
    denumerator = NP.full(m, nan)

    t = NP.isfinite(d) & (d != -1)
    u = NP.isinf(d)
    v = t|u

    R[:,t] = KX[:,t] - NP.multiply(MX[:,t], d[t])
    R[:,u] = MX[:,u]
    del KX
    del MX

    numerator[v] = NP.sqrt( \
            abs(2*norms(R[:,v])**2 -
            (columnwise_dot_product(X[:,v], R[:,v]) / ns[v])**2 )
        )

    denumerator[t] = weighted_norm_average(k_F, m_F, d[t])
    denumerator[u] = m_F

    eta[v] = numerator[v] / denumerator[v]
    eta[numerator==0] = 0


    assert eta.shape == (m,)
    assert NP.isrealobj(eta)
    assert not NP.any( NP.isnan(eta) )
    assert NP.all( eta >= 0 )
    assert NP.all( eta <= 1+2*eps )

    eta[eta > 1] = 1

    return eta



def compute_condition_number(K, M, d, X):
    assert K.dtype == M.dtype
    assert K.dtype == d.dtype
    assert K.dtype == X.dtype
    assert X.shape[1] == d.size

    if not utils.is_hermitian(K) or not utils.is_hermitian(M):
        raise ValueError("Matrices must be Hermitian")

    if NP.any( NP.isinf(d) ) or NP.any( NP.iscomplex(d) ) or NP.any( d==-1 ):
        raise ValueError("Eigenvalues must be regular real finite")

    m = d.size

    if m == 0:
        return NP.full(m, NP.nan, dtype=d.dtype)

    k_F = frobenius_norm(K)
    m_F = frobenius_norm(M)

    numerator = weighted_norm_average(k_F, m_F, d)
    X = X / norms(X)
    denumerator = columnwise_dot_product(M*X, X)

    kappa = numerator / denumerator
    assert kappa.shape == d.shape
    assert ~NP.any( NP.isnan(kappa) )
    assert NP.all( kappa >= 0 )

    kappa[(0 <= kappa) & (kappa < 1)] = 1

    return kappa



def compute_errors(K, M, d, X, block_size=256):
    n = K.shape[0]
    m = d.size

    eta = NP.full_like(d, NP.nan)
    kappa = NP.full_like(d, NP.nan)

    for l in xrange(0, m, block_size):
        r = min(l+block_size, n)

        eta[l:r] = compute_backward_error(K, M, d[l:r], X[:,l:r])
        kappa[l:r] = compute_condition_number(K, M, d[l:r], X[:,l:r])

    delta = eta * kappa

    return eta, delta
