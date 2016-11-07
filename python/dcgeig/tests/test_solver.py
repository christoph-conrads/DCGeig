#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import unittest

import numbers

import numpy as NP
import numpy.matlib as ML

import scipy.linalg as SL
import scipy.sparse as SS

import dcgeig.binary_tree as binary_tree
import dcgeig.gallery as gallery
import dcgeig.linalg as linalg
import dcgeig.solver as solver



class Test_estimate_trace(unittest.TestCase):
    def test_simple(self):
        n = 3
        b = 2
        f = lambda x: x

        mean, std = solver.estimate_trace(f, n, b)

        self.assertEqual(mean, 3)
        self.assertTrue(std < mean)



class Test_estimate_eigenvalue_count(unittest.TestCase):
    def test_simple(self):
        n = 9
        K = SS.diags(NP.arange(1, n+1), dtype=NP.float64, format='csc')
        M = SS.identity(n)

        l = 0.75
        r = 2 * l
        d = 50
        b = 5

        mean, std = solver.estimate_eigenvalue_count(K, M, l, r, d, b)

        eps = 0.05
        self.assertTrue( abs(mean - 1) < eps )
        self.assertTrue( std < mean )



class Test_compute_search_space(unittest.TestCase):
    def test_leaf_node(self):
        n = 2

        node = binary_tree.make_leaf_node(n)

        K = SS.spdiags(1.0 * NP.arange(1,n+1), 0, n, n, format='csc')
        M = SS.identity(n, dtype=K.dtype, format='csc')
        n_s = 1
        n_s_min = 1

        S = solver.compute_search_space(node, K, M, n_s, n_s_min)

        self.assertEqual( S.shape[0], n )
        self.assertEqual( S.shape[1], n_s )

        eps = NP.finfo(K.dtype).eps
        self.assertTrue( abs(S[0,0]) >= 1-eps )
        self.assertTrue( abs(S[0,0]) <= 1+eps )
        self.assertTrue( abs(S[1,0]) <= eps )


    def test_recursion(self):
        n = 9

        left = binary_tree.make_leaf_node(n/2)
        right = binary_tree.make_leaf_node(n-n/2)
        node = binary_tree.make_internal_node(left, right, n)

        K = SS.spdiags(1.0 * NP.arange(1, n+1), 0, n, n, format='csc')
        M = SS.identity(n, dtype=K.dtype, format='csc')
        n_s = 2
        n_s_min = 1

        S = solver.compute_search_space(node, K, M, n_s, n_s_min)

        self.assertEqual( S.shape[0], n )
        self.assertEqual( S.shape[1], n_s )

        Q = linalg.orthogonalize(S)

        self.assertTrue( abs(Q[0,0]) > 0.99 )



    def test_FEM_Laplacian_2D(self):
        n1 = 4
        a = 1.0
        n2 = 5
        b = 1.25
        m = n1*n2

        left = binary_tree.make_leaf_node(m/2)
        right = binary_tree.make_leaf_node(m-m/2)
        node = binary_tree.make_internal_node(left, right, m)

        K, M = gallery.fem_laplacian_2D_rectangle(n1, a, n2, b)

        S = solver.compute_search_space(node, K, M, 4, 2)

        self.assertEqual( S.shape[0], K.shape[0] )
        self.assertEqual( S.shape[1], 4 )

        Q = linalg.orthogonalize(S)
        A = Q.H * K * Q
        B = Q.H * M * Q

        d_min = min( SL.eigvalsh(A, B, eigvals=(0,0)) )
        r = d_min / NP.pi**2

        self.assertTrue( r >= (1.0/a)**2 + (1.0/b)**2 )
        self.assertTrue( r < (1.0/a)**2 + (2.0/b)**2 )



class Test_compute_largest_eigenvalue(unittest.TestCase):
    def test_simple(self):
        n = 5
        m = 3

        K = SS.diags(NP.arange(1, n+1), dtype=NP.float64, format='lil')
        K[-2] = 0
        K = SS.csc_matrix(K)

        M = SS.identity(n, format='lil')
        M[-1] = 0
        M = SS.csc_matrix(M)

        S = ML.eye(n, m, dtype=K.dtype)
        tol = 1e-2

        d_max = solver.compute_largest_eigenvalue(K, M, S, tol=tol)

        self.assertIsInstance(d_max, numbers.Real)
        self.assertTrue( abs(d_max - 3) <= tol )


    # arpack does not work with 1x1 matrices
    def test_1by1(self):
        n = 3
        K = SS.spdiags(1.0*NP.arange(1,n+1), 0, n, n)
        M = SS.identity(n, dtype=K.dtype)
        v = ML.eye(n, 1)

        d = solver.compute_largest_eigenvalue(K, M, v)

        self.assertIsInstance(d, numbers.Real)
        self.assertEqual(d, 1)



if __name__ == '__main__':
    unittest.main()
