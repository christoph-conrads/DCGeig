#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import unittest

import dcgeig.multilevel_tools as tools
import dcgeig.utils as utils
from dcgeig.sparse_tools import Tree

import numpy as NP
import numpy.matlib as ML

import scipy.sparse as SS
import scipy.linalg as SL
import scipy.sparse.linalg as LA



class Test_get_submatrices(unittest.TestCase):
    def test_simple(self):
        n1 = 2
        n2 = 3
        n3 = 4
        n = n1 + n2 + n3

        dim_tree = Tree.make_internal_node( \
            Tree.make_leaf_node({'n': n1}),
            Tree.make_leaf_node({'n': n2}),
            {'n': n})

        A11 = n1 * NP.ones([n1,n1])
        A12 = NP.zeros([n1,n2])
        A22 = n2 * NP.ones([n2,n2])
        A33 = n3 * NP.ones([n3,n3])

        A13 = -5 * NP.ones([n1,n3])
        A23 = -7 * NP.ones([n2,n3])

        A = SS.bmat([
            [A11, A12, A13],
            [A12.T, A22, A23],
            [A13.T, A23.T, A33]], format='csc')

        B11, B22, B33 = tools.get_submatrices(A, dim_tree)

        self.assertTrue( NP.all(A11 == B11) )
        self.assertTrue( NP.all(A22 == B22) )
        self.assertTrue( NP.all(A33 == B33) )



class Test_solve_SLE(unittest.TestCase):
    def test_simple(self):
        n = 3
        A = 4*SS.identity(n)
        I = (2*NP.identity(n), True)
        tree = Tree.make_leaf_node({'n': n, 'cholesky_factor': I})

        b = NP.ones([n,1])
        x = tools.solve_SLE(tree, A, b)

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

        x = tools.solve_SLE(tree, A, b)

        self.assertEqual( x.dtype, dtype )
        self.assertEqual( x.shape, b.shape )

        def compute_backward_error(A, x, b):
            r = b - A*x
            return SL.norm(r,1) / (LA.norm(A,1) * SL.norm(x,1) + SL.norm(b,1))

        eta = compute_backward_error(A, x, b)
        self.assertTrue( eta <= NP.finfo(dtype).eps )



class Test_compute_schur_complement(unittest.TestCase):
    def test_simple(self):
        n = 3
        A = SS.identity(n, format='csc')
        tree = Tree.make_leaf_node({'n': n})

        sle_tree = tools.compute_schur_complement(A, tree)

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

        schur_tree = tools.compute_schur_complement(A, tree)

        self.assertTrue( hasattr(schur_tree, 'schur_complement') )
        self.assertTrue( hasattr(schur_tree.left_child, 'cholesky_factor') )
        self.assertTrue( hasattr(schur_tree.right_child, 'cholesky_factor') )

        b = NP.ones([n,1], dtype=dtype)
        x = tools.solve_SLE(schur_tree, A, b)

        self.assertEqual( x.dtype, dtype )
        self.assertEqual( x.shape, b.shape )

        def compute_backward_error(A, x, b):
            r = b - A*x
            return SL.norm(r,1) / (LA.norm(A,1) * SL.norm(x,1) + SL.norm(b,1))

        eta = compute_backward_error(A, x, b)
        self.assertTrue( eta <= NP.finfo(dtype).eps )



class Test_make_eigenpair_selector(unittest.TestCase):
    class Options:
        def __init__(self, lambda_c, c_s, n_s_min):
            self.lambda_c = lambda_c
            self.c_s = c_s
            self.n_s_min = n_s_min



    def test_simple(self):
        options = self.Options(10, 2, 0)
        m = 50
        d = NP.arange(m)
        delta = NP.zeros(m)

        select = tools.make_eigenpair_selector(options, level=1)
        t = select(d, delta)

        self.assertTrue( NP.any(t) )



    def test_min_size(self):
        n_s_min = 8
        options = self.Options(100, 1, n_s_min)

        d = NP.arange(101, 150)
        delta = NP.zeros(d.size)

        select = tools.make_eigenpair_selector(options, level=1)
        t = select(d, delta)

        self.assertTrue( NP.all(t[:n_s_min]) )


    def test_select_only_finite_values(self):
        n_s_min = 10
        options = self.Options(1, 1, n_s_min)

        d = NP.array([100, float('inf')])
        delta = NP.zeros(d.size)

        select = tools.make_eigenpair_selector(options, level=1)
        t = select(d, delta)

        self.assertTrue( t[0] )
        self.assertFalse( t[1] )


    def test_level_selection(self):
        n = 100
        options = self.Options(1, 2, 32)
        d = NP.arange(n, dtype=NP.float64)
        delta = NP.ones(d.size)

        f = tools.make_eigenpair_selector(options, level=1)
        t1 = f(d, delta)

        g = tools.make_eigenpair_selector(options, level=100)
        t100 = g(d, delta)

        self.assertTrue( NP.sum(t100) >= NP.sum(t1) )
        self.assertTrue( NP.all( (t1 & t100) == t1 ) )



class Test_rayleigh_ritz(unittest.TestCase):
    def test_simple(self):
        n = 5
        m = 2

        ds = NP.arange(1.0 * n)
        K = SS.spdiags(ds, 0, n, n, format='csc')
        M = SS.identity(n, format='lil')

        S = ML.zeros( [n,m] )
        S[2:4,:] = NP.array( [[1, 1], [1, 0]] )

        d, X, eta, delta = tools.rayleigh_ritz(K, M, S)

        eps = NP.finfo(d.dtype).eps
        self.assertTrue( NP.all(eta <= n * eps) )



if __name__ == '__main__':
    unittest.main()
