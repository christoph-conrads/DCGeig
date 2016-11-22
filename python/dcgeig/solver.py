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

import dcgeig.binary_tree as binary_tree
import dcgeig.error_analysis as error_analysis
import dcgeig.linalg as linalg
import dcgeig.options
import dcgeig.polynomial as polynomial
import dcgeig.sparse_tools as sparse_tools
import dcgeig.subspace_iteration as subspace_iteration
import dcgeig.utils as utils

import copy

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



# /size/ of a search space = dimension of the search space
def compute_search_space_sizes(n_s_min, lambda_1, lambda_c, d, b, node,K,M,n_s):
    assert isinstance(n_s_min, int)
    assert n_s_min >= 0
    assert isinstance(lambda_1, numbers.Real)
    assert lambda_1 > 0
    assert isinstance(lambda_c, numbers.Real)
    assert lambda_c > lambda_1
    assert isinstance(d, int)
    assert d > 0
    assert isinstance(b, int)
    assert b > 0
    assert isinstance(node, binary_tree.Node)
    assert SS.isspmatrix(K)
    assert SS.isspmatrix(M)
    assert isinstance(n_s, int)
    assert n_s > 0


    if n_s <= n_s_min or node.is_leaf_node():
        new_node = copy.copy(node)
        new_node.n_s = n_s

        new_node.left_child = None
        new_node.right_child = None

        return new_node


    K11, K22, _ = sparse_tools.get_submatrices_bisection(node, K)
    M11, M22, _ = sparse_tools.get_submatrices_bisection(node, M)

    mean1, std1 = estimate_eigenvalue_count(K11, M11, lambda_1, lambda_c, d, b)
    mean2, std2 = estimate_eigenvalue_count(K22, M22, lambda_1, lambda_c, d, b)

    n_l = mean1 + std1
    n_r = mean2 + std2

    c = n_s / (n_l + n_r)

    n_sl = int(NP.ceil(c * n_l))
    n_sr = int(NP.ceil(c * n_r))

    left = node.left_child
    right = node.right_child

    new_left = compute_search_space_sizes( \
            n_s_min, lambda_1, lambda_c, d, b, left, K11, M11, n_sl)
    new_right = compute_search_space_sizes( \
            n_s_min, lambda_1, lambda_c, d, b, right, K22, M22, n_sr)

    new_node = copy.copy(node)
    new_node.n_s = new_left.n_s + new_right.n_s
    new_node.left_child = new_left
    new_node.right_child = new_right

    assert new_node.n_s >= n_s

    return new_node



def compute_search_space(lambda_c, node, K, M):
    assert isinstance(lambda_c, numbers.Real)
    assert lambda_c > 0
    assert isinstance(node, binary_tree.Node)
    assert SS.isspmatrix(K)
    assert SS.isspmatrix(M)

    n_s = node.n_s
    assert isinstance(n_s, int)
    assert n_s > 0


    if node.is_leaf_node():
        d, X = linalg.rayleigh_ritz(K, M)
        assert n_s <= K.shape[0]/2

        n_c = NP.sum(d <= lambda_c)
        m = 2 * max(n_s, n_c)
        return d[:m], X[:,:m], NP.empty(0), NP.empty(0)


    K11, K22, _ = sparse_tools.get_submatrices_bisection(node, K)
    M11, M22, _ = sparse_tools.get_submatrices_bisection(node, M)

    left = node.left_child
    right = node.right_child

    d1, X1, _, _ = compute_search_space(lambda_c, left, K11, M11)
    d2, X2, _, _ = compute_search_space(lambda_c, right, K22, M22)

    del K11; del M11
    del K22; del M22

    # combine search spaces
    d = NP.concatenate([d1, d2])
    X = SL.block_diag(X1, X2)
    X = ML.matrix(X)

    del d1; del d2
    del X1; del X2
    assert d.size >= n_s

    LL = linalg.spll(K)

    n_c = NP.sum(d <= lambda_c)

    for k in range(10):
        subspace_iteration.minmax_chebyshev( \
                LL.solve, K, M, X, max(d), 2, overwrite_b=True)

        d, X = linalg.rayleigh_ritz(K, M, X)
        eta, delta = error_analysis.compute_errors(K, M, d, X)

        t = d <= lambda_c
        t[0] = True

        n_c = NP.sum(d <= lambda_c)

        fmt = 'CSS {:d}  {:6d} {:4d} {:4d}  {:8.2e} {:8.2e} {:8.2e}  {:8.2e} {:8.2e}'
        n = K.shape[0]
        print fmt.format( \
                k, n, d.size, n_c,
                NP.median(eta), NP.max(eta), NP.max(eta[t]),
                NP.min(d) / lambda_c, NP.max(d) / lambda_c)

        if max(eta[t]) < NP.finfo(NP.float32).eps:
            break

    n_c = NP.sum(d <= lambda_c)
    assert n_c <= n_s

    return d, X, eta, delta



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


        K = A[:,t][t,:]
        M = B[:,t][t,:]

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


        # compute partitioning
        G = sparse_tools.matrix_to_graph(K)
        root, perm = sparse_tools.multilevel_bisection(G, options.n_direct)
        iperm = NP.argsort(perm)

        K = K[:,perm][perm,:]
        M = M[:,perm][perm,:]

        del G


        # count eigenvalues
        sigma = lambda_c/s

        t0 = time.time()
        c0 = time.clock()

        mean, std = estimate_eigenvalue_count( \
            K+sigma*M, M, sigma, 2*sigma, poly_degree, n_trial)

        if mean+std < 0.5:
            return NP.ones(0), ML.ones([n,0]), NP.ones(0), NP.ones(0)

        n_s = int( NP.ceil(mean + std) )

        root = compute_search_space_sizes( \
                n_s_min, sigma, 2*sigma, poly_degree, n_trial, \
                root, K+sigma*M, M, n_s)

        c1 = time.clock()
        t1 = time.time()

        fmt = 'Estimated eigenvalue count: {:d} ({:.1f}s {:.1f}s)'
        show( fmt.format(n_s, t1-t0, c1-c0) )

        del t0; del t1
        del c0; del c1


        # compute search space
        t0 = time.time()
        c0 = time.clock()
        d, X, eta, delta = compute_search_space(lambda_c/s, root, K, M)
        c1 = time.clock()
        t1 = time.time()

        fmt = 'Search space computed ({:.1f}s {:.1f}s)'
        show( fmt.format(t1-t0, c1-c0) )

        del t0; del t1
        del c0; del c1


        # test for early return
        n_c = NP.sum(d <= lambda_c/s)

        if max(eta[:n_c]) <= eta_max and max(delta[:n_c]/d[:n_c]) <= delta_max:
            return s*d[:n_c], D*X[iperm,:n_c], eta[:n_c], s*delta[:n_c]


        # use subspace iterations for solutions
        t0 = time.time()
        c0 = time.clock()
        d, X, eta, delta = subspace_iteration.execute( \
                K, M, X, lambda_c/s, eta_max, delta_max)
        c1 = time.clock()
        t1 = time.time()

        fmt = 'Subspace iteration found {:d} eigenpairs ({:.1f}s {:.1f}s)'
        show( fmt.format(d.size, t1-t0, c1-c0) )

        del t0; del t1
        del c0; del c1

        return s*d, D*X[iperm,:], eta, s*delta


    rs = map( call_solve_gep, range(l) )

    return rs, labels
