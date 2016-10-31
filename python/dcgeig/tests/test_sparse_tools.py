#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import unittest

import numpy as NP
import numpy.matlib as ML
import numpy.random

import scipy.sparse as SS
import scipy.sparse.linalg as LA

import dcgeig.binary_tree as binary_tree
import dcgeig.sparse_tools as sparse_tools
import dcgeig.error_analysis as error_analysis



class Test_balance_matrix_pencil(unittest.TestCase):
    def test_nop(self):
        n = 3
        I = SS.identity(n)
        s, D = sparse_tools.balance_matrix_pencil(I, I)

        self.assertEqual( s, 1 )
        self.assertEqual( (D != I).nnz, 0 )



    def test_scale(self):
        n = 3
        I = SS.identity(n)
        s, D = sparse_tools.balance_matrix_pencil(2*I, I)

        self.assertEqual( s, 2 )
        self.assertEqual( (D != I).nnz, 0 )



    def test_diag(self):
        n = 4
        ks = NP.arange(n)
        K = SS.spdiags(ks, 0, n, n)
        M = SS.identity(n, format='lil')
        M[0,0] = 0
        s, D = sparse_tools.balance_matrix_pencil(K, M)

        D = SS.csc_matrix(D)
        self.assertEqual( D[0,0], 1 )



class Test_matrix_to_graph(unittest.TestCase):
    def check_return_value(self, G):
        self.assertTrue( SS.isspmatrix(G) )
        self.assertEqual( G.dtype, NP.float32 )
        self.assertEqual( (G < 0).nnz, 0 )


    def test_simple(self):
        n = 2
        K = SS.identity(n)

        G = sparse_tools.matrix_to_graph(K)
        self.check_return_value(G)


    def test_complex(self):
        n = 3

        K = SS.identity(n, dtype=NP.complex, format='lil')
        K[0,1] = -1.0j
        K[1,0] = +1.0j

        G = sparse_tools.matrix_to_graph(K)
        self.check_return_value(G)



class Test_multilevel_bisection(unittest.TestCase):
    def check_tree(self, n, tree):
        self.assertEqual( tree.n, n )

        if tree.is_leaf_node():
            return

        self.assertTrue( hasattr(tree, 'left_child') )
        self.assertTrue( hasattr(tree, 'right_child') )

        left = tree.left_child
        right = tree.right_child

        self.assertTrue( isinstance(left, binary_tree.BinaryTree) )
        self.assertTrue( isinstance(right, binary_tree.BinaryTree) )

        self.assertEqual( left.n + right.n, n )

        self.check_tree(tree.left_child.n, tree.left_child)
        self.check_tree(tree.right_child.n, tree.right_child)


    def check_permutation(self, n, perm):
        self.assertTrue( NP.all(NP.sort(perm) == NP.arange(n)) )


    def check_return_values(self, n, tree, perm):
        self.check_tree(n, tree)
        self.check_permutation(n, perm)



    def test_nop(self):
        n = 4
        A = SS.identity(n, dtype=NP.float32, format='csc')
        tree, perm = sparse_tools.multilevel_bisection(A, n)

        self.check_return_values(n, tree, perm)
        self.assertEqual( tree.get_height(), 0 )



    def test_2by2(self):
        n = 4
        A = SS.identity(n, dtype=NP.float32, format='csc')
        tree, perm = sparse_tools.multilevel_bisection(A, 1)

        self.check_return_values(n, tree, perm)
        self.assertEqual( tree.get_height(), 2 )



    def test_8by8(self):
        n = 8
        A = ML.matrix([
            [  0, 100,  10,  10,   1,   1,   1,   1],
            [100,   0,  10,  10,   1,   1,   1,   1],
            [ 10,  10,   0, 100,   1,   1,   1,   1],
            [ 10,  10, 100,   0,   1,   1,   1,   1],
            [  1,   1,   1,   1,   0, 100,  10,  10],
            [  1,   1,   1,   1, 100,   0,  10,  10],
            [  1,   1,   1,   1,  10,  10,   0, 100],
            [  1,   1,   1,   1,  10,  10, 100,   0]
        ], dtype=NP.float32)

        p = [0, 2, 4, 6, 7, 5, 3, 1]
        B = SS.csc_matrix(A[p,:][:,p])
        tree, perm = sparse_tools.multilevel_bisection(B, 2)

        self.check_return_values(n, tree, perm)
        self.assertTrue( tree.get_height(), 3 )

        C = B[:,perm][perm,:]
        self.assertEqual( LA.norm(C[:,:4][4:,:], 'fro')**2, 16 )
        self.assertEqual( LA.norm(C[:,:2][2:4,:], 'fro')**2, 400 )
        self.assertEqual( LA.norm(C[:,6:][4:6,:], 'fro')**2, 400 )



class Test_multilevel_nested_dissection(unittest.TestCase):
    def check_tree(self, n, tree):
        self.assertEqual( tree.n, n )

        if tree.is_leaf_node():
            return

        self.assertTrue( hasattr(tree, 'left_child') )
        self.assertTrue( hasattr(tree, 'right_child') )

        left = tree.left_child
        right = tree.right_child

        self.assertTrue( isinstance(left, binary_tree.BinaryTree) )
        self.assertTrue( isinstance(right, binary_tree.BinaryTree) )

        self.assertTrue( left.n + right.n <= n )

        self.check_tree(tree.left_child.n, tree.left_child)
        self.check_tree(tree.right_child.n, tree.right_child)



    def check_permutation(self, n, perm):
        self.assertTrue( NP.all(NP.sort(perm) == NP.arange(n)) )


    def check_return_values(self, n, tree, perm):
        self.check_tree(n, tree)
        self.check_permutation(n, perm)



    def test_nop(self):
        n = 4
        A = SS.identity(n)
        tree, perm = sparse_tools.multilevel_nested_dissection(A, n)

        self.check_return_values(n, tree, perm)
        self.assertEqual( tree.get_height(), 0 )



    def test_2by2(self):
        n = 4
        A = SS.identity(n)
        tree, perm = sparse_tools.multilevel_nested_dissection(A, 1)

        self.check_return_values(n, tree, perm)
        self.assertEqual( tree.get_height(), 2 )



    def test_multilevel(self):
        diag = lambda xs: SS.block_diag(xs, format='lil')

        A11 = diag( [7*NP.ones([7,7]), 6*NP.ones([6,6]), 3*NP.ones([3,3])] )
        A11[:,-1] = -1
        A11[-1,:] = -1

        A22 = diag( [5*NP.ones([5,5]), 4*NP.ones([4,4]), 2*NP.ones([2,2])] )
        A22[:,-1] = -2
        A22[-1,:] = -2

        A = diag( [A11, A22, NP.ones([1,1])] )
        A[:,-1] = -3
        A[-1,:] = -3

        n = A.shape[0]
        n_direct = 10
        self.assertEqual( n, 28 )

        p = numpy.random.permutation( NP.arange(n) )
        B = SS.csc_matrix(A[p,:][:,p])
        tree, perm = sparse_tools.multilevel_nested_dissection(B, n_direct)

        self.check_return_values(n, tree, perm)
        self.assertTrue( tree.get_height(), 3 )

        n_1 = tree.left_child.n
        n_2 = tree.right_child.n

        C = B[:,perm][perm,:].todense()
        self.assertEqual( n_1 + n_2, n-1 )
        self.assertEqual( C[-1,-1], -3 )

        self.check_tree(n_1, tree.left_child)
        self.check_tree(n_2, tree.right_child)

        self.assertTrue( NP.all(C[n_1-1,:n_1] < 0) )
        self.assertTrue( NP.all(C[:n_1,n_1-1] < 0) )
        self.assertTrue( C[n-2,n-2] < 0 )



class Test_rayleigh_ritz(unittest.TestCase):
    def test_simple(self):
        n = 5
        m = 2

        ds = NP.arange(1.0 * n)
        K = SS.spdiags(ds, 0, n, n, format='csc')
        M = SS.identity(n, format='lil')

        S = ML.zeros( [n,m] )
        S[2:4,:] = NP.array( [[1, 1], [1, 0]] )

        d, X = sparse_tools.rayleigh_ritz(K, M, S)
        eta, delta = error_analysis.compute_errors(K, M, d, X)

        eps = NP.finfo(d.dtype).eps
        self.assertTrue( NP.all(eta <= n * eps) )



if __name__ == '__main__':
    unittest.main()
