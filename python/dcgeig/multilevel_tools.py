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
    default.tol = NP.finfo(NP.float32).eps
    default.c_s = 10
    default.n_s_min = 32
    default.max_num_iterations = 10
    default.w = 0

    return default





def bound_expected_backward_error(options, K, K21, M, M21):
    assert K21.shape == M21.shape

    lambda_c = options.lambda_c
    assert NP.isrealobj(lambda_c)
    assert lambda_c > 0

    norm = lambda A: LA.norm(A, 'fro')
    p = max(K21.shape)

    k = norm(K)
    k21 = norm(K21)
    assert k21 <= k

    m = norm(M)
    m21 = norm(M21)
    assert m21 <= m

    bound = -1

    if m21 == 0 or k21/m21 >= k/m:
        bound = NP.sqrt(1.5 / p) * k21/k
    else:
        nominator = k21**2 + lambda_c**2 * m21**2
        denominator = k**2 + lambda_c**2 * m**2
        bound = NP.sqrt(1.5 / p) * NP.sqrt(nominator / denominator)

    assert NP.isrealobj(bound)
    assert bound >= 0
    assert bound <= 1

    return bound



def compute_errors(K, M, d, X, block_size=256):
    assert isinstance(block_size, int)
    assert block_size > 0

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



def make_eigenpair_selector(options, level):
    assert isinstance(level, int)
    assert level >= 0

    lambda_c = options.lambda_c
    c_s = options.c_s
    n_s_min = options.n_s_min

    assert lambda_c > 0
    assert c_s >= 1
    assert isinstance(n_s_min, int)
    assert n_s_min >= 0

    lambda_s = c_s * lambda_c if level==0 else level * c_s * lambda_c

    def f(d, delta):
        n_c = NP.sum(d <= lambda_c)
        m = min(max(n_s_min, 2*n_c), d.size)

        t = d <= lambda_s
        t[:m] = True

        return t & NP.isfinite(d)

    return f



def apply_selection(t, d, X, eta, delta):
    return d[t], X[:,t], eta[t], delta[t]



def make_termination_test(options, level):
    lambda_c = options.lambda_c

    def f(d, X, eta, delta):
        eps32 = NP.finfo(NP.float32).eps
        delta_rel = delta / abs(d)

        t = d <= lambda_c

        if ~NP.any(t):
            t[0] = True

        if level == 0:
            return NP.max(eta[t]) <= options.tol and NP.max(delta_rel[t]) <= 1
        else:
            return NP.max(eta[t]) <= eps32 and NP.max(delta_rel[t]) <= 1

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



def get_stats_header():
    fmt = ( \
        '{:>2s} {:3s} '
        '{:>6s} {:>4s} {:>4s}  '
        '{:>8s} {:>7s} {:>8s}  '
        '{:>8s} {:>7s} {:>8s}  '
        '{:>8s} {:>7s} {:>8s} '
        '{:>2s} {:>5s} {:>6s} {:>6s} {:>6s}  '
        '{:>4s} {:>4s} {:>4s}\n')

    header = fmt.format( \
        'id', 'lvl',
        'n', 'n_c', 'n_s',
        'min:ev', 'max:ev', 'median:ev',
        'min:be', 'max:be', 'median:be',
        'min:fe', 'max:fe', 'median:fe',
        'iter', 't-sle', 't-rr', 't-wc', 't-cpu',
        'mems', 'memd', 'mlu')

    return header



def make_stats_tree( \
        options, K, M, level, ptree,
        d, X, eta, delta,
        wallclock_time_start, cpu_time_start,
        wallclock_time_rr = None, wallclock_time_sle = None,
        num_iterations=0, LU=None,
        stats1=None, stats2=None,
        **kwargs):
    wallclock_time = time.time() - wallclock_time_start
    cpu_time = time.clock() - cpu_time_start

    if not hasattr(options, 'show_stats'):
        return


    # get memory consumption
    def get_nnz_LU(LU):
        if not LU:
            return 0
        if NP.all(LU.perm_c == LU.perm_r):
            return LU.nnz/2
        return LU.nnz

    dtype = d.dtype
    b = dtype.itemsize
    dynamic_memory_B = d.nbytes + X.nbytes + eta.nbytes + delta.nbytes
    static_memory_B = 3*b * K.nnz + 3*b * M.nnz # factor 3 for CSC, CSR matrix
    memory_LU_B = 3*b * get_nnz_LU(LU)


    # output
    line_fmt = (\
        '%3d %3d ' # id level
        '%6d %4d %4d  ' # n n_c n_s
        '%8.2e %8.2e %8.2e  ' # eigenvalue statistics
        '%8.2e %8.2e %8.2e  ' # backward error statistics
        '%8.2e %8.2e %8.2e  ' # relative forward error statistics
        '%2d %6.1f %6.1f %6.1f %6.1f  ' # num_iterations, timing information
        '%4.0f %4.0f %4.0f\n') # memory(static) memory(dynamic) in MB

    n = K.shape[0]
    s = options.s

    t = d <= options.lambda_c
    t[0] = True
    n_c = NP.sum(t)
    n_s = d.size

    rfe = delta / abs(d) # rfe = relative forward error

    wc_time_rr = wallclock_time_rr if wallclock_time_rr else wallclock_time
    wc_time_sle = wallclock_time_sle if wallclock_time_sle else 0

    dynamic_memory_MB = dynamic_memory_B / 1000.0**2
    static_memory_MB = static_memory_B / 1000.0**2
    memory_LU_MB = memory_LU_B / 1000.0**2

    line = line_fmt % (
        ptree.id, level,
        n, n_c, n_s,
        s*NP.min(d), s*NP.max(d), s*NP.median(d),
        NP.min(eta[t]), NP.max(eta[t]), NP.median(eta[t]),
        NP.min(rfe[t]), NP.max(rfe[t]), NP.median(rfe[t]),
        num_iterations, wc_time_sle, wc_time_rr, wallclock_time, cpu_time,
        static_memory_MB, dynamic_memory_MB, memory_LU_MB)

    options.show_stats(line)
