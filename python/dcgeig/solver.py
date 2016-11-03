#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import numbers

import copy

import numpy as NP
import numpy.random
import numpy.linalg as NL

import scipy.sparse as SS
import scipy.sparse.linalg as LA

import dcgeig.binary_tree as binary_tree
import dcgeig.gallery as gallery
import dcgeig.polynomial as polynomial
import dcgeig.sparse_tools as sparse_tools

import time



def estimate_trace(f, b, k, eps, P, node):
    assert callable(f)
    assert isinstance(b, int)
    assert b > 1 # greater one or the variance is zero after the first estimate
    assert isinstance(k, int)
    assert k > 0
    assert isinstance(eps, numbers.Real)
    assert eps > 0
    assert eps < 1
    assert SS.isspmatrix(P)
    assert P.shape[0] >= P.shape[1]
    assert P.shape[1] == node.n


    if not node.is_leaf_node():
        left = node.left_child
        right = node.right_child

        n = node.n
        n1 = left.n
        n2 = right.n
        assert n1 + n2 == n

        new_left = estimate_trace(f, b, k, eps, P[:,:n1], left)
        new_right = estimate_trace(f, b, k, eps, P[:,n1:], right)

        trace_mean = new_left.trace_mean + new_right.trace_mean
        trace_std = NL.norm([new_left.trace_std, new_right.trace_std])

        new_node = binary_tree.make_internal_node(new_left, new_right, n)
        new_node.trace_mean = trace_mean
        new_node.trace_std = trace_std

        return new_node


    n = node.n
    mean = 0.0
    std = 1.0
    xs = NP.empty(0)

    random = numpy.random.RandomState(seed=1)

    while k*std > eps * mean:
        V = 2 * random.randint(0, 2, [n,b]).astype(P.dtype) - 1
        W = P.H * f(P*V)
        xs = NP.append( xs, NP.einsum('ij,ij->j', V, W, casting='no') )

        mean = NP.mean(xs)
        std = NP.std(xs, ddof=1)


    new_node = copy.copy(node)
    new_node.trace_mean = mean
    new_node.trace_std = std

    return new_node



def estimate_eigenvalue_count(node, K, M, lambda_1, lambda_c, d, k, eps, b=32):
    assert isinstance(node, binary_tree.Node)
    assert SS.isspmatrix(K)
    assert SS.isspmatrix(M)
    assert isinstance(lambda_1, numbers.Real)
    assert lambda_1 > 0
    assert isinstance(lambda_c, numbers.Real)
    assert lambda_c > lambda_1
    assert isinstance(d, int)
    assert d > 0
    assert isinstance(k, int)
    assert k > 0
    assert isinstance(eps, numbers.Real)
    assert eps > 0
    assert eps < 1
    assert isinstance(b, int)
    assert b > 1


    n = node.n
    f = polynomial.approximate_projection(d, lambda_1, lambda_c, K, M)
    P = SS.identity(n, format='csc')

    new_node = estimate_trace(f, b, k, eps, P, node)

    assert isinstance(new_node, binary_tree.Node)

    return new_node



def compute_smallest_eigenvalue(node, K, M, tol=0.0):
    assert isinstance(node, binary_tree.Node)
    assert SS.isspmatrix(K)
    assert SS.isspmatrix(M)
    assert isinstance(tol, numbers.Real)
    assert tol < 1


    def f(v0=None):
        n = node.n
        v0 = NP.ones([n,1]) if v0 is None else v0
        assert v0.shape[0] == n
        assert v0.shape[1] == 1

        ds, xs = LA.eigsh(K, M=M, k=1, which='SM', tol=tol, v0=v0)

        return ds[0], xs


    if node.is_leaf_node():
        d, x = f()

        new_node = copy.copy(node)
        new_node.smallest_eigenvalue = d

        return new_node, d, x


    left = node.left_child
    right = node.right_child

    n = node.n
    n1 = left.n
    n2 = right.n

    K11, K22, _ = sparse_tools.get_submatrices_bisection(K, n1, n2)
    M11, M22, _ = sparse_tools.get_submatrices_bisection(M, n1, n2)

    new_left, _, x1 = compute_smallest_eigenvalue(left, K11, M11, 1e-2)
    new_right, _, x2 = compute_smallest_eigenvalue(right, K22, M22, 1e-2)

    assert x1.shape[0] == n1
    assert x1.shape[1] == 1
    assert x2.shape[0] == n2
    assert x2.shape[1] == 1

    v0 = NP.vstack([x1, x2])
    d, x = f(v0)

    new_node = binary_tree.make_internal_node(new_left, new_right, n)
    new_node.smallest_eigenvalue = d

    return new_node, d, x



if __name__ == '__main__':
    n = 80
    K, M = gallery.fem_laplacian_2D(n)

    A = sparse_tools.matrix_to_graph(K)
    root, perm = sparse_tools.multilevel_bisection(A, n_direct=1024)

    K = K[:,perm][perm,:]
    M = M[:,perm][perm,:]

    t0 = time.clock()
    new_root, d, x = compute_smallest_eigenvalue(root, K, M)
    t1 = time.clock()
    print '{:10s} {:8.1f}s'.format('cse', t1-t0)

    v0 = NP.ones([K.shape[0],1])

    del d
    del x
    del t0
    del t1

    t0 = time.clock()
    ds, x = LA.eigsh(K, M=M, k=1, which='SM', v0=v0)
    t1 = time.clock()

    print '{:10s} {:8.1f}s'.format('arpack', t1-t0)
