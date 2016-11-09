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
import scipy.sparse.linalg as LA

import dcgeig.error_analysis as error_analysis
import dcgeig.linalg as linalg
import dcgeig.options
import dcgeig.polynomial as polynomial
import dcgeig.sparse_tools as sparse_tools
import dcgeig.subspace_iteration as subspace_iteration
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



def compute_search_space(node, K, M, n_s, n_s_min):
    assert SS.isspmatrix(K)
    assert SS.isspmatrix(M)
    assert isinstance(n_s, int)
    assert n_s > 0
    assert K.shape[0] > n_s
    assert isinstance(n_s_min, int)
    assert n_s_min > 0


    if n_s <= n_s_min or node.is_leaf_node():
        LL = linalg.spll(K)
        solve = LA.aslinearoperator(LL)

        n = K.shape[0]
        v0 = NP.ones([n,1], dtype=K.dtype)
        d, X = LA.eigsh(K, M=M, k=n_s, sigma=0, OPinv=solve, v0=v0, tol=1e-1)

        return ML.matrix(X)


    K11, K22, _ = sparse_tools.get_submatrices_bisection(node, K)
    M11, M22, _ = sparse_tools.get_submatrices_bisection(node, M)

    S1 = compute_search_space(node.left_child, K11, M11, n_s/2, n_s_min)
    S2 = compute_search_space(node.right_child, K22, M22, n_s-n_s/2, n_s_min)

    # compute largest ev
    d1_max = linalg.compute_largest_eigenvalue(K11, M11, S1, tol=1e-1)
    d2_max = linalg.compute_largest_eigenvalue(K22, M22, S2, tol=1e-1)
    d_max = max(d1_max, d2_max)

    del K11; del M11
    del K22; del M22


    # combine search spaces
    linalg.orthogonalize(S1, do_overwrite=True)
    linalg.orthogonalize(S2, do_overwrite=True)
    S = SL.block_diag(S1, S2)
    S = ML.matrix(S)

    del S1; del S2

    LL = linalg.spll(K)
    solve = LL.solve
    subspace_iteration.inverse_iteration( \
        solve, K, M, S, 2*d_max, overwrite_b=True)

    return S



def execute(options, A, B, lambda_c):
    assert isinstance(options, dcgeig.options.Options)
    assert utils.is_hermitian(A)
    assert utils.is_hermitian(B)
    assert A.dtype == B.dtype
    assert isinstance(lambda_c, numbers.Real)
    assert lambda_c > 0

    n = A.shape[0]
    n_direct = options.n_direct
    n_s_min = options.n_s_min
    eta_max = options.eta_max
    delta_max = options.delta_max

    l, labels = sparse_tools.get_subproblems(A, B)

    def call_solve_gep(i):
        t = labels == i


        if NP.sum(t) < n_direct:
            K = A[:,t][t,:]
            M = B[:,t][t,:]

            d, X = linalg.rayleigh_ritz(K, M)
            eta, delta = error_analysis.compute_errors(K, M, d, X)

            return d, X, eta, delta


        M = B[:,t][t,:]
        K = A[:,t][t,:] + lambda_c * M

        # balance matrix pencil
        s, D = sparse_tools.balance_matrix_pencil(K, M)
        K = SS.csc_matrix(D * K * D)
        M = SS.csc_matrix(D * (s*M) * D)

        # compute partitioning
        G = sparse_tools.matrix_to_graph(K)
        root, perm = sparse_tools.multilevel_bisection(G, options.n_direct)

        K = K[:,perm][perm,:]
        M = M[:,perm][perm,:]

        del G

        # count eigenvalues
        mean, std = estimate_eigenvalue_count(K,M,lambda_c/s,2*lambda_c/s,50,50)
        n_s = int( NP.ceil(mean + std) )

        # compute search space
        S = compute_search_space(root, K, M, n_s, n_s_min)
        S[perm,:] = S

        # use subspace iterations for solutions
        K = A[:,t][t,:]
        M = B[:,t][t,:]

        s, D = sparse_tools.balance_matrix_pencil(K, M)
        K = SS.csc_matrix(D * K * D)
        M = SS.csc_matrix(D * (s*M) * D)

        LL = linalg.spll(K)

        d, X, eta, delta = subspace_iteration.execute( \
                LL.solve, K, M, S, lambda_c/s, eta_max, delta_max)

        return s*d, X, eta, s*delta


    rs = map( call_solve_gep, range(l) )

    return rs, labels
