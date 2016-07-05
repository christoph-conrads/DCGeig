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
import numpy.linalg

import scipy
import scipy.sparse as SS
import scipy.linalg as SL
import scipy.sparse.linalg as LA

import copy

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

    return default



def get_submatrices(A, ptree):
    assert SS.isspmatrix_csc(A)

    assert isinstance(ptree, Tree)
    assert ptree.left_child
    assert ptree.right_child
    assert ptree.n == A.shape[0]

    n1 = ptree.left_child.n
    n2 = ptree.right_child.n
    n3 = ptree.n - n1 - n2
    assert n3 >= 0

    # avoid using n3 in range because it may be zero, e.g.,
    # do not use A33 = A[:,-n3:][-n3:,:]
    A11 = A[:,:n1][:n1,:]
    A22 = A[:,n1:n1+n2][n1:n1+n2,:]
    A33 = A[:,n1+n2:][n1+n2:,:]

    return A11, A22, A33




# Solve with schur complement
def solve_SLE(tree, A, B):
    assert SS.isspmatrix(A)
    assert isinstance(tree, Tree)
    assert B.shape[0] == A.shape[0]


    if Tree.is_leaf_node(tree):
        assert hasattr(tree, 'cholesky_factor')

        C = tree.cholesky_factor
        X = SL.cho_solve(C, B)

        assert X.dtype == B.dtype
        assert NP.all(X.shape == B.shape)

        return X


    # get submatrices
    left = tree.left_child
    right = tree.right_child

    n1 = left.n
    n2 = right.n
    n3 = A.shape[0] - n1 - n2

    A11, A22, A33 = get_submatrices(A, tree); A33 = None
    B1 = B[:n1,:]
    B2 = B[n1:n1+n2,:]

    U1 = solve_SLE(left, A11, B1)
    U2 = solve_SLE(right, A22, B2)
    U = NP.vstack([U1, U2])

    # manual memory management
    B1 = None; B2 = None


    # return in block diagonal case
    if n3 == 0:
        assert U.dtype == B.dtype
        assert NP.all(U.shape == B.shape)

        return U


    # full complement
    assert hasattr(tree, 'schur_complement')

    B3 = B[-n3:,:]
    A31 = A[:,:-n3][-n3:,:]
    S = tree.schur_complement

    X3 = SL.cho_solve(S, B3 - A31 * U)

    # manual memory management
    U = None; A31 = None; B3 = None

    A13 = A[:,-n3:][:n1,:].todense()
    A23 = A[:,-n3:][n1:n1+n2,:].todense()

    H1 = ML.matrix( solve_SLE(left, A11, A13) )
    H2 = ML.matrix( solve_SLE(right, A22, A23) )

    # manual memory management
    A11 = None; A22 = None; A13 = None; A23 = None

    X1 = U1 - H1 * X3
    X2 = U2 - H2 * X3
    X = NP.vstack([X1, X2, X3])

    assert X.dtype == B.dtype
    assert NP.all(X.shape == B.shape)

    return X



def compute_schur_complement(A, tree):
    assert SS.isspmatrix(A)
    assert isinstance(tree, Tree)


    if Tree.is_leaf_node(tree):
        assert not hasattr(tree, 'cholesky_factor')

        new = copy.copy(tree)
        new.cholesky_factor = scipy.linalg.cho_factor(A.todense(), lower=True)

        return new


    # get submatrices
    left = tree.left_child
    right = tree.right_child

    n1 = left.n
    n2 = right.n
    n3 = A.shape[0] - n1 - n2

    A11, A22, A33 = get_submatrices(A, tree)

    new_left = compute_schur_complement(A11, left)
    new_right = compute_schur_complement(A22, right)

    # block diagonal case?
    if n3 == 0:
        new_tree = copy.copy(tree)
        new_tree.left_child = new_left
        new_tree.right_child = new_right

        return new_tree


    # matrix not block diagonal
    A13 = A[:,-n3:][:n1,:].todense()
    A23 = A[:,-n3:][n1:n1+n2,:].todense()

    T13 = solve_SLE(new_left, A11, A13)
    T23 = solve_SLE(new_right,A22, A23)
    T = NP.vstack([T13, T23])

    A3x = A[:,:-n3][-n3:,:]
    S = A33 - A3x * T
    C = SL.cho_factor(S, lower=True)

    new_tree = copy.copy(tree)
    new_tree.left_child = new_left
    new_tree.right_child = new_right
    new_tree.schur_complement = C

    return new_tree



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



def compute_errors(K, M, d, X):
    eta = EA.compute_backward_error(K, M, d, X)
    kappa = EA.compute_condition_number(K, M, d, X)
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
        '{:>2s} {:>6s} {:>7s} {:>7s} {:>8s}  '
        '{:>4s} {:>4s}\n')

    header = fmt.format( \
        'id', 'lvl',
        'n', 'n_c', 'n_s',
        'min:ev', 'max:ev', 'median:ev',
        'min:be', 'max:be', 'median:be',
        'min:fe', 'max:fe', 'median:fe',
        'iter', 't-sle', 't-rr', 't-wc', 't-cpu',
        'mems', 'memd')

    return header



def make_stats_tree( \
        options, K, M, level, ptree,
        d, X, eta, delta,
        wallclock_time_start, cpu_time_start,
        wallclock_time_rr=None, wallclock_time_sle=None,
        num_iterations=0,
        stats1=None, stats2=None,
        **kwargs):
    wallclock_time = time.time() - wallclock_time_start
    cpu_time = time.clock() - cpu_time_start

    if not hasattr(options, 'show_stats'):
        return


    # estimate memory consumption
    def get_memory_schur_decomposition(ptree):
        if Tree.is_leaf_node(ptree):
            return ptree.cholesky_factor[0].nbytes

        n1 = ptree.left_child.n
        n2 = ptree.right_child.n
        n3 = ptree.n - n1 - n2
        if n3 > 0:
            return ptree.schur_complement[0].nbytes

        return 0


    dtype = d.dtype
    b = dtype.itemsize
    dynamic_memory_B = d.nbytes + X.nbytes + eta.nbytes + delta.nbytes
    # factor 3 for CSC, CSR matrix
    static_memory_B = \
        3*b * K.nnz + 3*b * M.nnz + get_memory_schur_decomposition(ptree)


    # output
    line_fmt = (\
        '%3d %3d ' # id level
        '%6d %4d %4d  ' # n n_c n_s
        '%8.2e %8.2e %8.2e  ' # eigenvalue statistics
        '%8.2e %8.2e %8.2e  ' # backward error statistics
        '%8.2e %8.2e %8.2e  ' # relative forward error statistics
        '%2d %7.1f %7.1f %7.1f %8.1f  ' # num_iterations, timing information
        '%4.0f %4.0f\n') # memory(static) memory(dynamic) in MB

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

    line = line_fmt % (
        ptree.id, level,
        n, n_c, n_s,
        s*NP.min(d), s*NP.max(d), s*NP.median(d),
        NP.min(eta[t]), NP.max(eta[t]), NP.median(eta[t]),
        NP.min(rfe[t]), NP.max(rfe[t]), NP.median(rfe[t]),
        num_iterations, wc_time_sle, wc_time_rr, wallclock_time, cpu_time,
        static_memory_MB, dynamic_memory_MB)

    options.show_stats(line)
