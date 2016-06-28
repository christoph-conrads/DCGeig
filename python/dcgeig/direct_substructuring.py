#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

from dcgeig.sparse_tools import Tree

import numpy as NP
import numpy.matlib as ML

import scipy.sparse as SS
import scipy.linalg as SL

import copy


def solve_SLE(tree, A, B, overwrite_b=False):
    assert isinstance(tree, Tree)
    assert SS.isspmatrix(A)
    assert B.shape[0] == A.shape[0]

    if overwrite_b:
        solve_SLE_overwrite(tree, A, B)
    else:
        return solve_SLE_impl(tree, A, B)



def solve_SLE_impl(tree, A, B):
    assert isinstance(tree, Tree)
    assert SS.isspmatrix(A)
    assert B.shape[0] == A.shape[0]


    if Tree.is_leaf_node(tree):
        assert hasattr(tree, 'cholesky_factor')

        C = tree.cholesky_factor
        X = SL.cho_solve(C, B, check_finite=False)

        assert X.dtype == B.dtype
        assert NP.all(X.shape == B.shape)

        return X


    # get submatrices
    left = tree.left_child
    right = tree.right_child

    n = A.shape[0]
    n1 = left.n
    n2 = right.n
    n3 = n - n1 - n2

    A11 = A[:,:n1][:n1,:]
    A22 = A[:,n1:n1+n2][n1:n1+n2,:]

    X = NP.empty_like(B)
    X[:n1,:] = solve_SLE(left, A11, B[:n1,:])
    X[n1:n1+n2,:] = solve_SLE(right, A22, B[n1:n1+n2,:])


    # return in block diagonal case
    if n3 == 0:
        assert X.dtype == B.dtype
        assert NP.all(X.shape == B.shape)

        return X


    # full complement
    assert hasattr(tree, 'schur_complement')

    S = tree.schur_complement
    X[-n3:,:] = SL.cho_solve( \
                S, B[-n3:,:]-A[:,:-n3][-n3:,:]*X[:-n3,:], check_finite=False)
    del S

    H = ML.empty([n1+n2,n3], dtype=A.dtype)
    H[:n1,:] = solve_SLE(left, A11, A[:,-n3:][:n1,:].todense())
    H[n1:,:] = solve_SLE(right, A22, A[:,-n3:][n1:n1+n2,:].todense())

    del A11; del A22

    X[:-n3,:] -= H*X[-n3:,:]

    assert X.dtype == B.dtype
    assert NP.all(X.shape == B.shape)

    return X



def solve_SLE_overwrite(tree, A, B):
    assert isinstance(tree, Tree)
    assert SS.isspmatrix(A)
    assert B.shape[0] == A.shape[0]


    if Tree.is_leaf_node(tree):
        assert hasattr(tree, 'cholesky_factor')

        C = tree.cholesky_factor
        B[:,:] = SL.cho_solve(C, B, check_finite=False)

        return


    # get submatrices
    left = tree.left_child
    right = tree.right_child

    n = A.shape[0]
    n1 = left.n
    n2 = right.n
    n3 = n - n1 - n2

    A11 = A[:,:n1][:n1,:]
    A22 = A[:,n1:n1+n2][n1:n1+n2,:]

    solve_SLE_overwrite(left, A11, B[:n1,:])
    solve_SLE_overwrite(right, A22, B[n1:n1+n2,:])


    # return in block diagonal case
    if n3 == 0:
        return


    # full complement
    assert hasattr(tree, 'schur_complement')

    S = tree.schur_complement
    B[-n3:,:] -= A[:,:-n3][-n3:,:] * B[:-n3,:]
    B[-n3:,:] = SL.cho_solve(S, B[-n3:,:], check_finite=False)
    del S

    H = ML.empty([n1+n2,n3], dtype=A.dtype)
    H[:n1,:] = solve_SLE(left, A11, A[:,-n3:][:n1,:].todense())
    H[n1:,:] = solve_SLE(right, A22, A[:,-n3:][n1:n1+n2,:].todense())

    del A11; del A22

    B[:-n3,:] -= H*B[-n3:,:]



def setup(tree, A):
    assert isinstance(tree, Tree)
    assert SS.isspmatrix(A)


    if Tree.is_leaf_node(tree):
        assert not hasattr(tree, 'cholesky_factor')

        new = copy.copy(tree)
        new.cholesky_factor = SL.cho_factor(A.todense(), lower=True)

        return new


    # get submatrices
    left = tree.left_child
    right = tree.right_child

    n = A.shape[0]
    n1 = left.n
    n2 = right.n
    n3 = n - n1 - n2

    A11 = A[:,:n1][:n1,:]
    A22 = A[:,n1:n1+n2][n1:n1+n2,:]

    new_left = setup(left, A11)
    new_right = setup(right, A22)


    # block diagonal case?
    if n3 == 0:
        new_tree = copy.copy(tree)
        new_tree.left_child = new_left
        new_tree.right_child = new_right

        return new_tree


    # matrix not block diagonal
    A13 = A[:,-n3:][:n1,:].todense()
    A23 = A[:,-n3:][n1:n1+n2,:].todense()

    T = ML.empty( [n1+n2,n3], dtype=A.dtype )
    T[:n1,:] = solve_SLE(new_left, A11, A13)
    T[n1:,:] = solve_SLE(new_right,A22, A23)

    A33 = A[:,-n3:][-n3:,:]
    S = A33 - A[:,:-n3][-n3:,:] * T
    C = SL.cho_factor(S, lower=True, check_finite=False)

    new_tree = copy.copy(tree)
    new_tree.left_child = new_left
    new_tree.right_child = new_right
    new_tree.schur_complement = C

    return new_tree
