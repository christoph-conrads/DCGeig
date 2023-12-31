#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numbers

import numpy as NP
import numpy.linalg
import numpy.matlib as ML

import scipy.linalg
import scipy.sparse as SS
import scipy.sparse.csgraph
import scipy.sparse.linalg as LA

import dcgeig
import dcgeig.binary_tree as binary_tree
import dcgeig.metis as metis
import dcgeig.utils as utils



def balance_matrix_pencil(K, M):
    if not utils.is_hermitian(K):
        raise ValueError('K must be Hermitian')
    if not utils.is_hermitian(M):
        raise ValueError('M must be Hermitian')
    if K.shape[0] != M.shape[0]:
        raise ValueError('Matrices must have the same dimension')


    def get_nearest_power_of_2(xs):
        ys = NP.round( NP.log2(xs) )
        zs = NP.power(2, ys)
        return zs


    normK_F = LA.norm(K, 'fro')
    normM_F = LA.norm(M, 'fro')
    s_raw = normK_F / normM_F if normM_F != 0 else 1
    s = get_nearest_power_of_2(s_raw)

    A = K
    B = s*M

    norms_K = LA.norm(A, axis=0)
    norms_M = LA.norm(B, axis=0)
    norms = numpy.linalg.norm( NP.stack([norms_K, norms_M]), ord=2, axis=0 )

    n = A.shape[0]
    max_norm = max(norms)
    eps = NP.finfo(norms.dtype).eps

    t = norms >= n * eps * max_norm
    rs = NP.full_like(norms, 1)
    rs[t] = get_nearest_power_of_2(max_norm / norms[t])

    D = SS.spdiags(NP.sqrt(rs), 0, n, n)

    # check return values
    assert isinstance(s, numbers.Real)
    assert s > 0
    assert NP.isrealobj(rs)
    assert NP.all(rs > 0)

    return s, D



def matrix_to_graph(A, p=2):
    if not SS.isspmatrix(A):
        raise ValueError('K must be a sparse matrix')
    if not utils.is_hermitian(A):
        raise ValueError('K must be Hermitian')

    G = SS.csc_matrix(abs(A), dtype=NP.float32, copy=True)
    G.data = abs(G.data)**p

    assert NP.all(G.data >= 0)
    assert not NP.any( NP.isinf(G.data) )

    return G



def multilevel_bisection(A, n_direct):
    if not SS.isspmatrix_csc(A):
        raise ValueError('A must be a CSC matrix')
    if not utils.is_hermitian(A):
        raise ValueError('A must be symmetric')
    if (not type(n_direct) is int) or (n_direct <= 0):
        raise ValueError('n_direct must be a positive integer')

    n = A.shape[0]

    if n <= n_direct:
        tree = binary_tree.make_leaf_node(n)
        perm = NP.arange(n)
        return tree, perm

    t = metis.bisection(A)

    A_1 = A[:,t]
    A11 = A_1[t,:]
    A_2 = A[:,~t]
    A22 = A_2[~t,:]

    left_child, perm11 = multilevel_bisection(A11, n_direct)
    right_child, perm22 = multilevel_bisection(A22, n_direct)

    assert NP.sum(t) == perm11.shape[0]
    assert NP.sum(~t) == perm22.shape[0]
    assert left_child.n + right_child.n == n

    tree = binary_tree.make_internal_node(left_child, right_child, n)

    p = NP.arange(n)
    p1 = p[t]
    p2 = p[~t]
    perm = NP.concatenate( [p1[perm11], p2[perm22]] )

    return tree, perm



def multilevel_nested_dissection(A, n_direct):
    if not SS.isspmatrix(A):
        raise ValueError('A must be a sparse matrix')
    if min(A.shape) == 0:
        raise ValueError('A must not be empty')
    if not NP.issubdtype(A.dtype, NP.float):
        raise ValueError('A must be real')
    if not utils.is_hermitian(A):
        raise ValueError('A must be symmetric')
    if not isinstance(n_direct, int) or (n_direct <= 0):
        raise ValueError('n_direct must be a positive integer')

    A = SS.csc_matrix(A)
    n = A.shape[0]

    if n <= n_direct:
        perm = NP.arange(n)
        sizes = NP.array([n/2, n-n/2, 0])
        tree = binary_tree.make_leaf_node(n)
        return tree, perm


    perm, sizes = metis.nested_dissection(A)
    assert perm.shape == (n,)
    assert sizes.shape == (3,)
    assert NP.sum(sizes) == n

    n_1 = sizes[0]
    n_2 = sizes[1]
    n_3 = sizes[2]

    p_1 = perm[:n_1]
    p_2 = perm[n_1:n_1+n_2]
    p_3 = perm[n_1+n_2:]

    B11 = A[:,p_1][p_1,:]
    B22 = A[:,p_2][p_2,:]

    B21 = A[:,p_1][p_2,:]
    assert B21.nnz == 0

    left_sizes, q_1 = multilevel_nested_dissection(B11, n_direct)
    right_sizes, q_2 = multilevel_nested_dissection(B22, n_direct)

    perm_ret = NP.concatenate( [p_1[q_1], p_2[q_2], p_3] )
    assert NP.all( NP.sort(perm_ret) == NP.arange(n) )

    tree = binary_tree.make_internal_node(left_sizes, right_sizes, n)

    return tree, perm_ret



def get_submatrices_bisection(node, A):
    assert isinstance(node, binary_tree.Node)
    assert not node.is_leaf_node()
    assert SS.isspmatrix_csc(A)
    assert utils.is_hermitian(A)

    n = node.n
    n1 = node.left_child.n
    n2 = node.right_child.n

    assert n == A.shape[0]
    assert n == n1 + n2

    A_1 = A[:,:n1]
    A11 = A_1[:n1,:]
    A12 = A_1[n1:,:]
    A22 = A[:,n1:][n1:,:]

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
    l_H, labels_H = scipy.sparse.csgraph.connected_components(H, directed=False)

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
