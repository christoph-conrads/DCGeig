#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest

import numbers

import numpy as NP
import numpy.matlib as ML

import scipy.linalg as SL
import scipy.sparse as SS

import dcgeig.binary_tree as binary_tree
import dcgeig.error_analysis as error_analysis
import dcgeig.gallery as gallery
import dcgeig.linalg as linalg
import dcgeig.options
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



class Test_estimate_search_space_sizes(unittest.TestCase):
    def test_simple(self):
        n = 5
        K = SS.spdiags(1.0 * NP.arange(1,n+1), 0, n, n, format='csc')
        M = SS.identity(n)

        n_s = 2

        node = binary_tree.make_leaf_node(n)
        new_node = \
            solver.estimate_search_space_sizes(30, 20, 4, 2.5, node, K, M, n_s)

        self.assertTrue( abs(new_node.n_s_mean - n_s) <= 0.1 )
        self.assertTrue( new_node.n_s_std < 0.01 )



    def test_2levels(self):
        n = 8
        h = 1.0 / (n+1)
        lambda_c = 2/h**2 * (1 - NP.cos(3*NP.pi/(n+1)))

        M = SS.identity(n**2, format='csc')
        K = gallery.fdm_laplacian([n,n], [1.0,1.0])
        d, X = gallery.fdm_laplacian_eigenpairs([n,n], [1.0,1.0])

        left = binary_tree.make_leaf_node(n**2/2)
        right = binary_tree.make_leaf_node(n**2/2)
        node = binary_tree.make_internal_node(left, right, n**2)

        new_node = solver.estimate_search_space_sizes( \
                2, 20, 10, lambda_c, node, K, M, 4)

        self.assertIsInstance(new_node, binary_tree.Node)
        self.assertIsInstance(new_node.n_s_mean, numbers.Real)
        self.assertEqual( NP.round(new_node.n_s_mean), 4)

        self.assertIsInstance(new_node.left_child.n_s_mean, numbers.Real)
        self.assertEqual( NP.round(new_node.left_child.n_s_mean), 2 )

        self.assertIsInstance(new_node.right_child.n_s_mean, numbers.Real)
        self.assertEqual( NP.round(new_node.right_child.n_s_mean), 2 )



class Test_compute_search_space(unittest.TestCase):
    def test_leaf_node(self):
        n = 3

        node = binary_tree.make_leaf_node(n)
        node.n_s = 1

        K = SS.spdiags(1.0 * NP.arange(1,n+1), 0, n, n, format='csc')
        M = SS.identity(n, dtype=K.dtype, format='csc')
        lambda_c = 0.5

        d, S = solver.compute_search_space(0.1, lambda_c, node, K, M)

        self.assertIsInstance( d, NP.ndarray )
        self.assertIsInstance( S, ML.matrix )
        self.assertEqual( S.shape[0], n )
        self.assertTrue( S.shape[1] >= node.n_s )

        m = S.shape[1]
        eps = NP.finfo(K.dtype).eps
        e1 = NP.eye(n, 1)
        self.assertTrue( abs(min(d) - 1) <= eps)
        self.assertTrue( SL.norm(S.H * S - ML.eye(m)) <= m*eps )
        self.assertTrue( SL.norm(NP.dot(e1.T, S)) > 1-eps )


    def test_recursion(self):
        n = 9

        left = binary_tree.make_leaf_node(n/2)
        left.n_s = 1
        right = binary_tree.make_leaf_node(n-n/2)
        right.n_s = 1
        node = binary_tree.make_internal_node(left, right, n)
        node.n_s = 2

        K = SS.spdiags(1.0 * NP.arange(1, n+1), 0, n, n, format='csc')
        M = SS.identity(n, dtype=K.dtype, format='csc')

        tol = 0.1
        lambda_c = 1.0

        d, S = solver.compute_search_space(tol, lambda_c, node, K, M)

        self.assertIsInstance( d, NP.ndarray )
        self.assertEqual( S.shape[0], n )
        self.assertTrue( S.shape[1] >= node.n_s )
        self.assertTrue( S.shape[1] <= 2*node.n_s )

        m = S.shape[1]
        eps = NP.finfo(S.dtype).eps
        self.assertTrue( SL.norm(S.H * S - ML.eye(m)) <= m*eps )

        e1 = ML.eye(n, 1)
        self.assertTrue( SL.norm(e1.H * S) > 0.99 )



    def test_FEM_Laplacian_2D(self):
        n1 = 4
        a = 1.0
        n2 = 5
        b = 1.25
        m = n1*n2

        left = binary_tree.make_leaf_node(m/2)
        left.n_s = 2
        right = binary_tree.make_leaf_node(m-m/2)
        right.n_s = 2
        node = binary_tree.make_internal_node(left, right, m)
        node.n_s = 4

        K, M = gallery.fem_laplacian([n1,n2], [a,b])

        tol = 0.1
        lambda_c = 2 * NP.pi**2

        d, S = solver.compute_search_space(tol, lambda_c, node, K, M)

        self.assertIsInstance( d, NP.ndarray )
        self.assertEqual( S.shape[0], K.shape[0] )
        self.assertTrue( S.shape[1] >= node.n_s )
        self.assertTrue( S.shape[1] <= 2*node.n_s )

        m = S.shape[1]
        eps = NP.finfo(S.dtype).eps
        self.assertTrue( SL.norm(S.H * S - ML.eye(m)) <= m*eps )

        A = S.H * K * S
        B = S.H * M * S

        d_min = min( SL.eigvalsh(A, B, eigvals=(0,0)) )
        r = d_min / NP.pi**2

        self.assertTrue( r >= (1.0/a)**2 + (1.0/b)**2 )
        self.assertTrue( r < (1.0/a)**2 + (2.0/b)**2 )



class Test_execute(unittest.TestCase):
    def test_simple(self):
        n = 6
        dtype = NP.float64

        M = SS.identity(n, dtype=dtype, format='csc')
        K = SS.csc_matrix( \
                NP.array([ \
                    [50, 1, 1, 1, 1, 1],
                    [ 1,20,-1, 0, 0, 0],
                    [ 1,-1,20,-1, 0, 0],
                    [ 1, 0,-1,20, 0, 0],
                    [ 1, 0, 0, 0,10, 1],
                    [ 1, 0, 0, 0, 1,10]], dtype=dtype))

        options = dcgeig.options.Options()
        options.delta_max = 1.0

        rs, labels = solver.execute(options, K, M, 1e-8)

        self.assertEqual( len(rs), 1 )
        self.assertTrue( NP.all(labels == 0) )
        self.assertEqual( labels.size, n )

        d = rs[0][0]
        X = rs[0][1]
        eta = rs[0][2]
        delta = rs[0][3]

        self.assertIsInstance(d, NP.ndarray)
        self.assertIsInstance(X, NP.ndarray)
        self.assertEqual( X.shape[0], n )

        self.assertTrue( NP.all(eta < 2*NP.finfo(dtype).eps) )
        self.assertTrue( NP.all(delta < options.delta_max) )


    def test_revert_balancing(self):
        n = 12
        h = 1.0 / (n+1)

        A = gallery.fdm_laplacian_1D(n)
        B = SS.identity(n)

        D = SS.identity(n, format='lil')
        D[0,0] = 8
        D[1,1] = 128

        K = SS.csc_matrix(D * A * D)
        M = SS.csc_matrix(D * B * D)

        options = dcgeig.options.Options()
        options.eta_max = 1e-10
        options.n_s_min = 1
        options.n_direct = 11
        options.num_trial_vectors = 5

        eigvals = 2 / h**2 * (1 - NP.cos(NP.pi * NP.arange(1,n+1) / (n+1)))
        eigvals = NP.sort(eigvals)
        lambda_c = (eigvals[0] + eigvals[1]) / 2
        d0 = min(eigvals)
        x0 = NP.sin(NP.pi * NP.arange(1,n+1) / (n+1) ).reshape([n,1])
        y0 = SL.solve(D.todense(), x0)

        rs, labels = solver.execute(options, K, M, lambda_c)

        self.assertEqual( len(rs), 1 )
        self.assertEqual( labels.size, n )

        l = labels[0]
        d = rs[l][0]
        X = rs[l][1]

        self.assertIsInstance(d, NP.ndarray)
        self.assertIsInstance(X, NP.ndarray)
        self.assertEqual( d.size, 1 )
        self.assertEqual( X.shape[0], n )

        eps32 = NP.finfo(NP.float32).eps
        self.assertTrue( (d - d0) <= eps32 )
        self.assertTrue( SL.norm(X / X[0,0] - y0 / y0[0]) <= eps32 )



    def test_zero_estimate(self):
        n = 4

        K = 1e6 * gallery.fdm_laplacian_1D(n)
        M = SS.identity(n, format='csc')

        options = dcgeig.options.Options()
        options.num_trial_vectors = 2
        options.n_direct = 2

        rs, labels = solver.execute(options, K, M, 1e-8)

        self.assertEqual( len(rs), 1 )
        self.assertEqual( rs[0][0].size, 0 )



    def test_fem_laplacian_2D(self):
        n1 = 10
        a = 1.0
        n2 = 15
        b = 1.5
        K, M = gallery.fem_laplacian([n1,n2], [a,b])

        options = dcgeig.options.Options()
        options.n_direct = 50
        options.n_s_min = 1
        options.num_trial_vectors = 24

        sigma = NP.pi**2 * ( (2.0/a)**2 + (3.0/b)**2) + 1
        rs, labels = solver.execute(options, K, M, sigma)

        self.assertEqual( len(rs), 1 )
        self.assertTrue( NP.all(labels == 0) )
        self.assertEqual( labels.size, n1*n2 )

        d = rs[0][0]
        X = rs[0][1]

        self.assertIsInstance(d, NP.ndarray)
        self.assertIsInstance(X, NP.ndarray)
        self.assertEqual( X.shape[0], n1*n2 )

        eta, delta = error_analysis.compute_errors(K, M, d, X)

        self.assertTrue( NP.all(eta <= options.eta_max) )
        self.assertTrue( NP.all(delta <= options.delta_max) )
        self.assertTrue( NP.all(eta/rs[0][2] <= 1.25) )
        self.assertTrue( NP.all(delta/rs[0][3] <= 1.25) )



if __name__ == '__main__':
    unittest.main()
