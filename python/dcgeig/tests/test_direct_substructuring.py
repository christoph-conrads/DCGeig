#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import unittest

import dcgeig.direct_substructuring as DS
import dcgeig.utils as utils
from dcgeig.sparse_tools import Tree

import numpy as NP
import numpy.matlib as ML

import scipy.sparse as SS
import scipy.linalg as SL
import scipy.sparse.linalg as LA



class Test_solve_SLE(unittest.TestCase):
    def test_simple(self):
        n = 3
        A = 4*SS.identity(n)
        I = (2*NP.identity(n), True)
        tree = Tree.make_leaf_node({'n': n, 'cholesky_factor': I})

        b = NP.ones([n,1])
        x = DS.solve_SLE(tree, A, b)

        self.assertTrue( NP.all(A*x == b) )



    def test_2(self):
        dtype = NP.float32

        n1 = 4
        L1 = ML.matrix( NP.tril(NP.ones([n1,n1], dtype=dtype)) )
        L1[1,0] = 10
        A1 = utils.force_hermiticity(L1*L1.T)
        C1 = (L1, True)
        left = Tree.make_leaf_node({'n': n1, 'cholesky_factor': C1})

        n2 = 2
        L2 = ML.matrix( NP.array([[4, 0], [1, 2]], dtype=dtype) )
        A2 = utils.force_hermiticity(L2*L2.T)
        C2 = (L2, True)
        right = Tree.make_leaf_node({'n': n2, 'cholesky_factor': C2})

        n3 = 3
        Ax3 = NP.eye(n3, n1+n2, dtype=dtype)

        L3 = ML.matrix(NP.array([[10,0,0], [-1,10,0], [0,-1,10]], dtype=dtype))
        LLT = utils.force_hermiticity(L3 * L3.T)
        A3 = LLT + NP.dot(Ax3, SL.solve(SL.block_diag(A1, A2), Ax3.T))
        A3 = utils.force_hermiticity( ML.matrix(A3) )
        S = (L3, True)

        n = n1 + n2 + n3
        node_data = {'n': n, 'schur_complement': S}
        tree = Tree.make_internal_node(left, right, node_data)

        A = SS.bmat([ \
                [SL.block_diag(A1, A2), Ax3.T],
                [Ax3, A3]], format='csc')
        b = NP.ones([n,1], dtype=dtype)

        x = DS.solve_SLE(tree, A, b)

        self.assertEqual( x.dtype, dtype )
        self.assertEqual( x.shape, b.shape )

        def compute_backward_error(A, x, b):
            r = b - A*x
            return SL.norm(r,1) / (LA.norm(A,1) * SL.norm(x,1) + SL.norm(b,1))

        eta = compute_backward_error(A, x, b)
        self.assertTrue( eta <= NP.finfo(dtype).eps )



class Test_setup(unittest.TestCase):
    def test_simple(self):
        n = 3
        A = SS.identity(n, format='csc')
        tree = Tree.make_leaf_node({'n': n})

        sle_tree = DS.setup(tree, A)

        self.assertTrue( isinstance(sle_tree, Tree) )
        self.assertTrue( hasattr(sle_tree, 'cholesky_factor') )

        L = sle_tree.cholesky_factor[0]
        self.assertTrue( NP.all(L*L.T == A) )



    def test_9by9(self):
        dtype = NP.float32

        n1 = 4
        L1 = ML.matrix( NP.tril(NP.ones([n1,n1], dtype=dtype)) )
        A1 = utils.force_hermiticity(L1*L1.T)
        left = Tree.make_leaf_node({'n': n1})

        n2 = 2
        L2 = ML.matrix( NP.array([[4, 0], [1, 2]], dtype=dtype) )
        A2 = utils.force_hermiticity(L2*L2.T)
        right = Tree.make_leaf_node({'n': n2})

        n3 = 3
        Ax3 = NP.eye(n3, n1+n2, dtype=dtype)

        L3 = ML.matrix(NP.array([[10,0,0], [-1,10,0], [0,-1,10]], dtype=dtype))
        LLT = utils.force_hermiticity(L3 * L3.T)
        A3 = LLT + NP.dot(Ax3, SL.solve(SL.block_diag(A1, A2), Ax3.T))
        A3 = utils.force_hermiticity( ML.matrix(A3) )

        n = n1 + n2 + n3
        tree = Tree.make_internal_node(left, right, {'n': n})

        A = SS.bmat([ \
                [SL.block_diag(A1, A2), Ax3.T],
                [Ax3, A3]], format='csc')

        schur_tree = DS.setup(tree, A)

        self.assertTrue( hasattr(schur_tree, 'schur_complement') )
        self.assertTrue( hasattr(schur_tree.left_child, 'cholesky_factor') )
        self.assertTrue( hasattr(schur_tree.right_child, 'cholesky_factor') )

        b = NP.ones([n,1], dtype=dtype)
        x = DS.solve_SLE(schur_tree, A, b)

        self.assertEqual( x.dtype, dtype )
        self.assertEqual( x.shape, b.shape )

        def compute_backward_error(A, x, b):
            r = b - A*x
            return SL.norm(r,1) / (LA.norm(A,1) * SL.norm(x,1) + SL.norm(b,1))

        eta = compute_backward_error(A, x, b)
        self.assertTrue( eta <= NP.finfo(dtype).eps )
