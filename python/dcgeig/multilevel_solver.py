#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import dcgeig.metis as metis
import dcgeig.utils as utils
import dcgeig.multilevel_tools as tools
import dcgeig.sparse_tools as sparse_tools
from dcgeig.sparse_tools import Tree

import numpy as NP
import numpy.matlib as ML

import scipy.linalg
import scipy.sparse as SS
import scipy.sparse.linalg as LA
import scipy.sparse.csgraph as graph

import numbers

import time



def get_submatrices(A, t):
    assert utils.is_hermitian(A)
    assert NP.any(t)
    assert NP.any(~t)

    A11 = A[:,t][t,:]
    A12 = A[:,~t][t,:]
    A22 = A[:,~t][~t,:]

    return A11, A22, A12



# This function finds all connected components in the graph induced by |K|+|M|
# and merges all nodes without edges into one component.
def get_subproblems(K, M):
    assert utils.is_hermitian(K)
    assert utils.is_hermitian(M)

    n = K.shape[0]
    assert n > 0


    # construct induced graph with diagonal entries set to zero
    # (no self-loops)
    U = abs(SS.triu(K, 1)) + abs(SS.triu(M, 1))
    G = U + U.T
    assert NP.isrealobj(G)


    # find disconnected nodes
    t = LA.norm(G, ord=float('inf'), axis=0) == 0


    if NP.all(t):
        return 1, NP.full(n, 0, dtype=int)


    # find connected components
    H = G[:,~t][~t,:]
    l_H, labels_H = graph.connected_components(H, directed=False)

    assert max(labels_H) < l_H

    # compute return values
    if NP.any(t):
        l = l_H + 1

        labels = NP.full( n, NP.nan, dtype=labels_H.dtype )
        labels[t] = l_H
        labels[~t] = labels_H
    else:
        l = l_H
        labels = labels_H

    assert l >= 1
    assert labels.size == n
    assert NP.all( labels >= 0 )
    assert NP.all( labels < l )

    return l, labels



def solve_gep(options, K, M, lambda_c, tol, level):
    assert utils.is_hermitian(K)
    assert utils.is_hermitian(M)
    assert K.dtype == M.dtype
    assert isinstance(lambda_c, numbers.Real)
    assert lambda_c > 0
    assert isinstance(tol, numbers.Real)
    assert tol > 0
    assert tol <= 1
    assert isinstance(level, int)
    assert level >= 0

    n = K.shape[0]

    assert n > 0


    select = tools.make_eigenpair_selector(options, lambda_c, level)


    # solve directly?
    if n <= options.n_direct:
        wallclock_time_start = time.time()
        cpu_time_start = time.clock()

        d, X, eta, delta = tools.rayleigh_ritz(K, M)

        t = select(d, delta)
        d, X, eta, delta = tools.apply_selection(t, d, X, eta, delta)

        return d, X, make_stats_tree(**locals())


    # divide
    G = sparse_tools.matrix_to_graph(K)
    t = metis.bisection(G)
    del G

    K11, K22, K12 = get_submatrices(K, t)
    M11, M22, M12 = get_submatrices(M, t)


    # conquer
    d1, X1, stats1 = solve_gep(options, K11, M11, lambda_c, tol, level+1)
    d2, X2, stats2 = solve_gep(options, K22, M22, lambda_c, tol, level+1)

    del K11; del K22
    del M11; del M22


    # combine
    wallclock_time_start = time.time()
    cpu_time_start = time.clock()

    d = NP.concatenate([d1, d2])
    X = ML.matrix( NP.full([n, d.size], 0, dtype=K.dtype) )
    X[ t, :d1.size] = X1
    X[~t, d1.size:] = X2

    del t
    del d1; del d2
    del X1; del X2

    # sort eigenpairs
    perm = NP.argsort(d)
    d = d[perm]
    X = X[:,perm]
    del perm


    eta, delta = tools.compute_errors(K, M, d, X)
    do_stop = tools.make_termination_test(options, lambda_c, level, tol)


    if do_stop(d, X, eta, delta):
        t = select(d, delta)
        d, X, eta, delta = tools.apply_selection(t, d, X, eta, delta)

        return d, X, make_stats_tree(**locals())


    if d.size == n:
        if level == 0:
            d, X, eta, delta = tools.rayleigh_ritz(K, M)

            t = select(d, delta)
            d, X, eta, delta = tools.apply_selection(t, d, X, eta, delta)

        return d, X, make_stats_tree(**locals())


    LU = LA.splu(K, diag_pivot_thresh=0)

    wallclock_time_sle = 0
    wallclock_time_rr = 0

    for i in range(1, options.max_num_iterations+1):
        wallclock_time_sle_start = time.time()
        t = eta > NP.finfo(K.dtype).eps
        tau = max(d)
        X[:,t] = LU.solve( (K - tau*M) * X[:,t] )
        wallclock_time_sle += time.time() - wallclock_time_sle_start

        wallclock_time_rr_start = time.time()
        d, X, eta, delta = tools.rayleigh_ritz(K, M, X)
        wallclock_time_rr += time.time() - wallclock_time_rr_start

        if do_stop(d, X, eta, delta):
            break

    assert not NP.any(NP.isinf(d))
    assert not NP.any(NP.isnan(d))

    t = select(d, delta)
    d, X, eta, delta = tools.apply_selection(t, d, X, eta, delta)

    return d, X, make_stats_tree(num_iterations=i, **locals())



def execute(options, A, B, lambda_c, tol, level=0):
    assert isinstance(options, tools.Options)
    assert utils.is_hermitian(A)
    assert utils.is_hermitian(B)
    assert A.dtype == B.dtype
    assert isinstance(lambda_c, numbers.Real)
    assert lambda_c > 0
    assert isinstance(tol, numbers.Real)
    assert tol > 0
    assert tol <= 1
    assert isinstance(level, int)
    assert level >= 0

    n = A.shape[0]
    l, labels = get_subproblems(A, B)

    def call_solve_gep(i):
        t = labels == i
        A_tt = A[:,t][t,:]
        B_tt = B[:,t][t,:]

        # balance matrix pencil
        s, D = sparse_tools.balance_matrix_pencil(A_tt, B_tt)
        K = SS.csc_matrix(D * A_tt * D)
        M = SS.csc_matrix(s * D * B_tt * D)

        d, X, stats = solve_gep(options, K, M, lambda_c/s, tol, level)

        return s*d, D*X, stats

    rs = map( call_solve_gep, range(l) )

    d = NP.concatenate( map(lambda t: t[0], rs) )
    xs = map(lambda t: t[1], rs)
    stats = map(lambda t: t[2], rs)

    del rs

    if l == 1:
        return d, xs[0], stats

    # assemble matrix of eigenvectors X
    X = NP.full( [n, d.size], 0, dtype=A.dtype )

    j = 0
    for i in range(l):
        t = labels == i
        m = xs[i].shape[1]
        X[t,j:j+m] = xs[i]
        j = j+m

    return d, X, stats



def make_stats_tree( \
        options, K, M, lambda_c, tol, level,
        d, X, eta, delta,
        wallclock_time_start, cpu_time_start,
        wallclock_time_rr=None, wallclock_time_sle=None,
        K12=None, M12=None,
        num_iterations=0, LU=None,
        stats1=None, stats2=None,
        **kwargs):
    assert LU is None or NP.all(LU.perm_c == LU.perm_r)

    wallclock_time = time.time() - wallclock_time_start
    cpu_time = time.clock() - cpu_time_start

    wc_time_rr = wallclock_time_rr if wallclock_time_rr else wallclock_time
    wc_time_sle = wallclock_time_sle if wallclock_time_sle else 0.0

    normK12 = NP.nan if K12 is None else LA.norm(K12)
    normM12 = NP.nan if M12 is None else LA.norm(M12)

    n = K.shape[0]
    n_c = NP.sum(d <= lambda_c)
    n_s = d.size
    nnz_LU = -1 if LU is None else LU.nnz

    rfe = delta / abs(d)

    data = {
            'n': n,
            'level': level,
            'nnz_K': K.nnz,
            'nnz_M': M.nnz,
            'norm_K': LA.norm(K),
            'norm_M': LA.norm(M),
            'norm_K12': normK12,
            'norm_M12': normM12,
            'nnz_LU': nnz_LU,
            'n_c': n_c,
            'n_s': n_s,
            'd_rel': d / lambda_c,
            'eta': eta,
            'rfe': rfe,
            'num_iterations': num_iterations,
            'wallclock_time_rr': wc_time_rr,
            'wallclock_time_sle': wc_time_sle,
            'wallclock_time': wallclock_time,
            'cpu_time': cpu_time
    }

    node = Tree.make_internal_node(stats1, stats2, data)
    return node
