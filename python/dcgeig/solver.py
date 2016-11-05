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
import numpy.linalg as LA

import scipy.sparse as SS

import dcgeig.binary_tree as binary_tree
import dcgeig.polynomial as polynomial
import dcgeig.sparse_tools as sparse_tools



def estimate_trace(f, b, P, node):
    assert callable(f)
    assert isinstance(b, int)
    assert b > 0
    assert SS.isspmatrix(P)
    assert isinstance(node, binary_tree.Node)
    assert P.shape[0] >= P.shape[1]
    assert P.shape[1] == node.n


    if not node.is_leaf_node():
        left = node.left_child
        right = node.right_child

        n = node.n
        n1 = left.n
        n2 = right.n
        assert n1 + n2 == n

        new_left = estimate_trace(f, b, P[:,:n1], left)
        new_right = estimate_trace(f, b, P[:,n1:], right)

        trace_mean = new_left.trace_mean + new_right.trace_mean
        trace_std = LA.norm([new_left.trace_std, new_right.trace_std])

        new_node = binary_tree.make_internal_node(new_left, new_right, n)
        new_node.trace_mean = trace_mean
        new_node.trace_std = trace_std

        return new_node


    n = node.n
    random = numpy.random.RandomState(seed=1)

    V = 2 * random.randint(0, 2, [n,b]).astype(P.dtype) - 1
    W = P.H * f(P*V)
    xs = NP.einsum('ij,ij->j', V, W, casting='no')

    mean = NP.mean(xs)
    std = NP.std(xs, ddof=1)

    new_node = copy.copy(node)
    new_node.trace_mean = mean
    new_node.trace_std = std

    return new_node



def estimate_eigenvalue_count(node, K, M, lambda_1, lambda_c, d, b):
    assert isinstance(node, binary_tree.Node)
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


    n = node.n
    f = polynomial.approximate_projection(d, lambda_1, lambda_c, K, M)
    P = SS.identity(n, format='csc')

    new_node = estimate_trace(f, b, P, node)

    assert isinstance(new_node, binary_tree.Node)

    return new_node
