#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import dcgeig
import dcgeig.error_analysis as EA
import dcgeig.utils as utils
from dcgeig.sparse_tools import Tree

import numpy as NP
import numpy.matlib as ML

import scipy.sparse as SS
import scipy.linalg
import scipy.sparse.linalg as LA

import time



class Options:
    pass



def get_default_options():
    default = Options()
    default.n_direct = 1024
    default.internal_tol = NP.finfo(NP.float32).eps
    default.c_s = 10
    default.n_s_min = 32
    default.max_num_iterations = 10

    return default



def compute_errors(K, M, d, X, block_size=256):
    n = K.shape[0]
    m = d.size

    eta = NP.full_like(d, NP.nan)
    kappa = NP.full_like(d, NP.nan)

    for l in xrange(0, m, block_size):
        r = min(l+block_size, n)

        eta[l:r] = EA.compute_backward_error(K, M, d[l:r], X[:,l:r])
        kappa[l:r] = EA.compute_condition_number(K, M, d[l:r], X[:,l:r])

    delta = eta * kappa

    return eta, delta



def make_eigenpair_selector(options, lambda_c, level):
    assert isinstance(level, int)
    assert lambda_c > 0
    assert level >= 0

    c_s = options.c_s
    n_s_min = options.n_s_min

    assert c_s >= 1
    assert isinstance(n_s_min, int)
    assert n_s_min >= 0

    lambda_s = c_s * lambda_c if level==0 else level * c_s * lambda_c

    def f(d):
        n_c = NP.sum(d <= lambda_c)
        m = min(max(n_s_min, 2*n_c), d.size)

        t = d <= lambda_s
        t[:m] = True

        return t & NP.isfinite(d)

    return f



def apply_selection(t, d, X, eta, delta):
    return d[t], X[:,t], eta[t], delta[t]



def make_termination_test(options, lambda_c, level, tol):
    assert lambda_c > 0
    assert NP.isrealobj(tol)
    assert tol > 0
    assert tol <= 1

    eps = tol if level == 0 else options.internal_tol

    def f(d, X, eta, delta):
        u = NP.finfo(X.dtype).eps
        delta_rel = delta / abs(d)

        t = d <= lambda_c

        if ~NP.any(t):
            t[0] = True

        return \
            (NP.max(eta[t]) <= eps and NP.max(delta_rel[t]) <= 1) or \
            (NP.max(eta[t]) <= u)

    return f



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

        Q, _ = scipy.linalg.qr(S, mode='economic')
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

    eta, delta = compute_errors(K, M, d, X)

    return d, X, eta, delta
