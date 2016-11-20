#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import numbers

import numpy as NP
import numpy.linalg as NL
import numpy.matlib as ML
import numpy.random

import scipy.linalg as SL
import scipy.sparse as SS

import dcgeig.error_analysis as error_analysis
import dcgeig.linalg as linalg
import dcgeig.options
import dcgeig.polynomial as polynomial
import dcgeig.sparse_tools as sparse_tools
import dcgeig.subspace_iteration as subspace_iteration
import dcgeig.utils as utils

import time



def estimate_trace(f, n, b, dtype=NP.float64):
    assert callable(f)
    assert isinstance(n, int)
    assert n > 0
    assert isinstance(b, int)
    assert b > 1
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



def compute_search_space(node, K, M, lambda_c, n_s_min, n_s):
    assert SS.isspmatrix(K)
    assert SS.isspmatrix(M)
    assert isinstance(lambda_c, numbers.Real)
    assert lambda_c > 0
    assert isinstance(n_s_min, int)
    assert n_s_min > 0
    assert n_s_min <= K.shape[0]
    assert isinstance(n_s, int)
    assert n_s >= 0


    if node.is_leaf_node() or 2*n_s >= K.shape[0]:
        d, X = linalg.rayleigh_ritz(K, M)

        n_c = NP.sum(d <= lambda_c)
        # 2*n_c <= n because 2*n_s >= n and n_s is at least as large as n_c
        n_t = max(n_s, 2*n_c, n_s_min)

        return d[:n_t], X[:,:n_t]


    K11, K22, _ = sparse_tools.get_submatrices_bisection(node, K)
    M11, M22, _ = sparse_tools.get_submatrices_bisection(node, M)

    left = node.left_child
    right = node.right_child

    d1, X1 = compute_search_space(left, K11, M11, lambda_c, n_s_min, n_s/2)
    d2, X2 = compute_search_space(right, K22, M22, lambda_c, n_s_min, n_s-n_s/2)

    del K11; del M11
    del K22; del M22

    # combine search spaces
    X = SL.block_diag(X1, X2)
    X = ML.matrix(X)

    d_max = max(max(d1), max(d2))

    del d1; del d2
    del X1; del X2

    LL = linalg.spll(K)

    for k in range(10):
        subspace_iteration.minmax_chebyshev( \
            LL.solve, K, M, X, d_max, 2, overwrite_b=True)

        d, X = linalg.rayleigh_ritz(K, M, X)
        eta, delta = error_analysis.compute_errors(K, M, d, X)

        d_max = max(d)

        t = d <= lambda_c
        t[0] = True

        fmt = 'CSS {:d}  {:6d} {:4d} {:4d} {:4d}  {:8.2e} {:8.2e} {:8.2e}  {:8.2e} {:8.2e}'
        n = K.shape[0]
        n_c = NP.sum(d <= lambda_c)
        print fmt.format( \
                k, n, n_s, d.size, n_c, NP.median(eta), NP.max(eta), NP.max(eta[t]), NP.min(d) / lambda_c,
                NP.max(d) / lambda_c)

        if max(eta[t]) < NP.finfo(NP.float32).eps:
            break

    n_c = NP.sum(d <= lambda_c)
    n_t = max(n_s, 2*n_c)

    return d[:n_t], X[:,:n_t]



def execute(options, A, B, lambda_c):
    assert isinstance(options, dcgeig.options.Options)
    assert utils.is_hermitian(A)
    assert utils.is_hermitian(B)
    assert A.dtype == B.dtype
    assert isinstance(lambda_c, numbers.Real)
    assert lambda_c > 0

    n_direct = options.n_direct
    n_s_min = options.n_s_min

    n_trial = options.num_trial_vectors
    poly_degree = options.polynomial_degree

    eta_max = options.eta_max
    delta_max = options.delta_max

    show = options.show

    l, labels = sparse_tools.get_subproblems(A, B)

    def call_solve_gep(i):
        t = labels == i
        n = NP.sum(t)


        if NP.sum(t) < n_direct:
            K = A[:,t][t,:]
            M = B[:,t][t,:]

            d, X = linalg.rayleigh_ritz(K, M)
            eta, delta = error_analysis.compute_errors(K, M, d, X)

            u = d-delta <= lambda_c

            return d[u], X[:,u], eta[u], delta[u]


        sigma = 5 * lambda_c
        M = B[:,t][t,:]
        K = A[:,t][t,:] + sigma * M

        # normalize matrix norm
        I = SS.identity(n) / NP.sqrt(n)
        s, _ = sparse_tools.balance_matrix_pencil(I, K)

        K = s * K
        M = s * M

        del s

        # balance matrix pencil
        s, D = sparse_tools.balance_matrix_pencil(K, M)
        K = SS.csc_matrix(D * K * D)
        M = SS.csc_matrix(D * (s*M) * D)

        # count eigenvalues
        t0 = time.time()
        c0 = time.clock()
        mean, std = estimate_eigenvalue_count( \
            K, M, sigma/s, 2*sigma/s, poly_degree, n_trial)
        c1 = time.clock()
        t1 = time.time()

        if mean+std < 0.5:
            return NP.ones(0), ML.ones([n,0]), NP.ones(0), NP.ones(0)

        n_s = int( NP.ceil(mean + std) )

        fmt = 'Estimated eigenvalue count: {:d} ({:.1f}s {:.1f}s)'
        show( fmt.format(n_s, t1-t0, c1-c0) )

        del t0; del t1
        del c0; del c1


        # compute partitioning
        K = A[:,t][t,:]
        M = B[:,t][t,:]

        s, D = sparse_tools.balance_matrix_pencil(K, M)
        K = SS.csc_matrix(D * K * D)
        M = SS.csc_matrix(D * (s*M) * D)

        G = sparse_tools.matrix_to_graph(K)
        root, perm = sparse_tools.multilevel_bisection(G, options.n_direct)

        K = K[:,perm][perm,:]
        M = M[:,perm][perm,:]

        del G

        # compute search space
        t0 = time.time()
        c0 = time.clock()
        d, X = compute_search_space(root, K, M, lambda_c/s, n_s_min, n_s)
        c1 = time.clock()
        t1 = time.time()

        iperm = NP.argsort(perm)
        X = X[iperm,:]

        fmt = 'Search space computed ({:.1f}s {:.1f}s)'
        show( fmt.format(t1-t0, c1-c0) )

        del t0; del t1
        del c0; del c1


        # use subspace iterations for solutions
        K = SS.csc_matrix(D * A[:,t][t,:] * D)
        M = SS.csc_matrix(D * (s*B[:,t][t,:]) * D)

        t0 = time.time()
        c0 = time.clock()
        d, X, eta, delta = subspace_iteration.execute( \
                K, M, X, lambda_c/s, sigma/s, eta_max, delta_max)
        c1 = time.clock()
        t1 = time.time()

        fmt = 'Subspace iteration found {:d} eigenpairs ({:.1f}s {:.1f}s)'
        show( fmt.format(d.size, t1-t0, c1-c0) )

        del t0; del t1
        del c0; del c1

        return s*d, D*X, eta, s*delta


    rs = map( call_solve_gep, range(l) )

    return rs, labels
