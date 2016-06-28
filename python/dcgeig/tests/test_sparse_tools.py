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

import dcgeig.sparse_tools as sparse_tools
from dcgeig.sparse_tools import Tree



class Test_Tree(unittest.TestCase):
    def test_simple(self):
        data = {'a': 1, 'b': 2.0, 'c': 'd'}
        tree = Tree.make_leaf_node(data)

        self.assertEqual( tree.a, data['a'] )
        self.assertEqual( tree.b, data['b'] )
        self.assertEqual( tree.c, data['c'] )


    def test_invalid(self):
        data = {1: '2'}

        with self.assertRaises(ValueError):
            tree = Tree.make_leaf_node(data)

        with self.assertRaises(ValueError):
            leaf = Tree.make_leaf_node({})
            tree = Tree.make_internal_node(leaf, leaf, data)



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



class Test_matrix_pencil_to_graph(unittest.TestCase):
    def check_return_value(self, G):
        self.assertTrue( SS.isspmatrix(G) )
        self.assertEqual( G.dtype, NP.float32 )
        self.assertEqual( (G < 0).nnz, 0 )


    def test_simple(self):
        n = 2
        K = SS.identity(n)
        M = SS.identity(n)
        w = 2

        G = sparse_tools.matrix_pencil_to_graph(K, M, w)
        self.check_return_value(G)



    def test_complex(self):
        n = 3

        K = SS.identity(n, dtype=NP.complex, format='lil')
        K[0,1] = -1.0j
        K[1,0] = +1.0j

        M = SS.identity(n, dtype=NP.complex, format='lil')
        M[0,2] = 3.0 + 4.0j
        M[2,0] = 3.0 - 4.0j

        w = 2
        G = sparse_tools.matrix_pencil_to_graph(K, M, w)

        self.check_return_value(G)
        self.assertEqual( G[0,1], 1 )
        self.assertEqual( G[1,0], 1 )
        self.assertEqual( G[0,2], 100 )
        self.assertEqual( G[2,0], 100 )



class Test_multilevel_bisection(unittest.TestCase):
    def check_tree(self, n, tree):
        self.assertEqual( tree.n, n )

        if Tree.is_leaf_node(tree):
            return

        self.assertTrue( hasattr(tree, 'left_child') )
        self.assertTrue( hasattr(tree, 'right_child') )

        left = tree.left_child
        right = tree.right_child

        self.assertTrue( isinstance(left, Tree) )
        self.assertTrue( isinstance(right, Tree) )

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
        self.assertEqual( Tree.get_height(tree), 0 )



    def test_2by2(self):
        n = 4
        A = SS.identity(n, dtype=NP.float32, format='csc')
        tree, perm = sparse_tools.multilevel_bisection(A, 1)

        self.check_return_values(n, tree, perm)
        self.assertEqual( Tree.get_height(tree), 2 )



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
        self.assertTrue( Tree.get_height(tree), 3 )

        C = B[:,perm][perm,:]
        self.assertEqual( LA.norm(C[:,:4][4:,:], 'fro')**2, 16 )
        self.assertEqual( LA.norm(C[:,:2][2:4,:], 'fro')**2, 400 )
        self.assertEqual( LA.norm(C[:,6:][4:6,:], 'fro')**2, 400 )



class Test_multilevel_nested_dissection(unittest.TestCase):
    def check_tree(self, n, tree):
        self.assertEqual( tree.n, n )

        if Tree.is_leaf_node(tree):
            return

        self.assertTrue( hasattr(tree, 'left_child') )
        self.assertTrue( hasattr(tree, 'right_child') )

        left = tree.left_child
        right = tree.right_child

        self.assertTrue( isinstance(left, Tree) )
        self.assertTrue( isinstance(right, Tree) )

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
        self.assertEqual( Tree.get_height(tree), 0 )



    def test_2by2(self):
        n = 4
        A = SS.identity(n)
        tree, perm = sparse_tools.multilevel_nested_dissection(A, 1)

        self.check_return_values(n, tree, perm)
        self.assertEqual( Tree.get_height(tree), 2 )



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
        self.assertTrue( Tree.get_height(tree), 3 )

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



class Test_add_postorder_id(unittest.TestCase):
    def check_tree(self, tree):
        self.assertTrue( isinstance(tree.id, int) )

        if Tree.is_leaf_node(tree):
            return

        self.assertTrue( hasattr(tree, 'left_child') )
        self.assertTrue( hasattr(tree, 'right_child') )

        left = tree.left_child
        right = tree.right_child

        self.assertTrue( isinstance(left, Tree) )
        self.assertTrue( isinstance(right, Tree) )

        self.assertTrue( left.id < right.id )
        self.assertEqual( right.id + 1, tree.id )

        self.check_tree(tree.left_child)
        self.check_tree(tree.right_child)


    def test_simple(self):
        self.check_tree( Tree.make_leaf_node({'id': -123}) )


    def test_unbalanced(self):
        t0 = Tree.make_leaf_node({'id': 100})
        t1 = Tree.make_leaf_node({'id': 101})
        t2 = Tree.make_internal_node(t0, t1, {'id': 102})
        t3 = Tree.make_leaf_node({'id': 103})
        t4 = Tree.make_internal_node(t2, t3, {'id': 104})

        self.check_tree(t4)


if __name__ == '__main__':
    unittest.main()
