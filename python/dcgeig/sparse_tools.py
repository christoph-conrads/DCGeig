#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import numbers

import numpy as NP
import numpy.linalg
import scipy.sparse as SS
import scipy.sparse.linalg as LA

import dcgeig.utils as utils
import dcgeig.metis as metis

import copy


class Tree:
    @staticmethod
    def make_leaf_node(data):
        return Tree(None, None, data)

    @staticmethod
    def make_internal_node(left_child, right_child, data):
        return Tree(left_child, right_child, data)


    def __init__(self, left_child, right_child, data):
        if not isinstance(data, dict):
            raise TypeError('data must be a dictionary')

        # this test does not catch strings that are not identifiers like '1'
        if filter(lambda k: not isinstance(k, str), data.keys()):
            raise ValueError('data keys must be strings')

        if 'left_child' in data or 'right_child' in data:
            raise AttributeError('Illegal keys found')

        self.left_child = left_child
        self.right_child = right_child
        self.__dict__.update(data)


    def has_left_child(self):
        return self.left_child is not None

    def has_right_child(self):
        return self.right_child is not None

    def is_leaf_node(self):
        return (not self.left_child) and (not self.right_child)


    def get_height(self):
        if Tree.is_leaf_node(self):
            return 0

        assert Tree.has_left_child(self)
        assert Tree.has_right_child(self)

        left = self.left_child
        right = self.right_child

        return max(Tree.get_height(left), Tree.get_height(right)) + 1



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
        tree = Tree.make_leaf_node({'n': n})
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

    tree = Tree.make_internal_node(left_sizes, right_sizes, {'n': n})

    return tree, perm_ret



def add_postorder_id(tree, sid=1):
    assert isinstance(tree, Tree)
    assert not hasattr(tree, 'id')

    if Tree.is_leaf_node(tree):
        new_tree = copy.copy(tree)
        new_tree.id = sid
        return new_tree

    new_left = add_postorder_id(tree.left_child, sid)
    new_right= add_postorder_id(tree.right_child, new_left.id+1)

    new_tree = copy.copy(tree)
    new_tree.left_child = new_left
    new_tree.right_child = new_right
    new_tree.id = new_right.id + 1

    return new_tree
