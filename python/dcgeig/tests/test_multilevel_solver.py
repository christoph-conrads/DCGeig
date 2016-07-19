#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import unittest

import dcgeig.error_analysis as EA
import dcgeig.utils as utils
import dcgeig.multilevel_solver as MS
import dcgeig.multilevel_tools as tools
import dcgeig.sparse_tools as sparse_tools
from dcgeig.sparse_tools import Tree

import numpy as NP
import scipy.sparse as SS

import numbers



class Test_get_submatrices(unittest.TestCase):
    def test_simple(self):
        n = 5
        ds = NP.arange(n, dtype=NP.complex64)
        A = SS.spdiags(ds, 0, n, n, format='lil')
        A[0,3] = 1+2j
        A[0,4] = 1+2j
        A[3,0] = 1-2j
        A[4,0] = 1-2j

        t = NP.array( [0, 1, 1, 0, 1], dtype=bool )
        A11, A22, A12 = MS.get_submatrices(SS.csc_matrix(A), t)

        self.assertEqual( A11.shape[0], 3 )
        self.assertEqual( A22.shape[0], 2 )

        B = SS.bmat( [[A11, A12], [A12.H, A22]] )

        u = [1, 2, 4, 0, 3]
        C = A[:,u][u,:]
        self.assertEqual( (C-B).nnz, 0 )



class Test_get_subproblems(unittest.TestCase):
    def test_simple(self):
        K = SS.csc_matrix([
            [1, 0, 0, 0, 0, 0],
            [0, 2, 1, 0, 0, 0],
            [0, 1, 3, 0, 0, 0],
            [0, 0, 0, 4, 0, 0],
            [0, 0, 0, 0, 5, 0],
            [0, 0, 0, 0, 0, 6]
        ], dtype=NP.complex64)
        M = SS.csc_matrix([
            [1, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0],
            [0, 0, 0, 4, 0-1j, 0],
            [0, 0, 0, 0+1j, 5, 0],
            [0, 0, 0, 0, 0, 6]
        ], dtype=NP.complex64)

        l, labels = MS.get_subproblems(K, M)

        self.assertEqual( l, 3 )
        self.assertEqual( labels[0], labels[-1] )
        self.assertEqual( labels[1], labels[2] )
        self.assertEqual( labels[3], labels[4] )
        self.assertEqual( labels.size, K.shape[0] )


    def test_diagonal(self):
        n = 3
        A = SS.identity(n, dtype=NP.complex128)

        l, labels = MS.get_subproblems(A, A)

        self.assertEqual( l, 1 )
        self.assertTrue( NP.all(labels == 0) )
        self.assertEqual( labels.size, n )



class Test_solve_gep(unittest.TestCase):
    def test_simple(self):
        n = 2
        K = SS.identity(n, format='csc')
        M = SS.identity(n, format='csc')
        options = tools.get_default_options()

        d, X, _ = MS.solve_gep(options, K, M, lambda_c=1.0, tol=1e-6, level=0)

        self.assertEqual( d[0], 1 )
        self.assertEqual( d[1], 1 )



    def test_returns_only_finite_eigenvalues(self):
        n = 2
        K = SS.identity(n, format='csc')
        M = SS.spdiags([0,1.0], 0, n, n, format='csc')
        options = tools.get_default_options()

        d, X, _ = MS.solve_gep(options, K, M, lambda_c=1, tol=1e-6, level=0)

        self.assertEqual( d.size, 1 )
        self.assertEqual( d[0], 1 )


    def test_multilevel(self):
        n = 3
        ds = 10 * (NP.arange(n, dtype=NP.float32) + 1)
        K = SS.spdiags(ds, 0, n, n, format='lil')
        K[0,1] = -1.0
        K[1,0] = -1.0
        K[1,2] = -1.0
        K[2,1] = -1.0

        M = SS.identity(n, dtype=NP.float32, format='csc')

        options = tools.get_default_options()
        options.n_direct = 2
        tol = 1e-6

        d, X, _ = MS.solve_gep(options, K, M, lambda_c=1, tol=tol, level=0)
        self.assertTrue( d.size > 0 )

        eta = EA.compute_backward_error(K, M, d, X)
        self.assertTrue( NP.all(eta < tol) )



class Test_execute(unittest.TestCase):
    def test_simple(self):
        n = 6
        dtype = NP.float32

        M = SS.identity(n, dtype=dtype, format='csc')
        K = SS.csc_matrix( \
                NP.array([ \
                    [50, 1, 1, 1, 1, 1],
                    [ 1,20,-1, 0, 0, 0],
                    [ 1,-1,20,-1, 0, 0],
                    [ 1, 0,-1,20, 0, 0],
                    [ 1, 0, 0, 0,10, 1],
                    [ 1, 0, 0, 0, 1,10]], dtype=dtype))

        options = tools.get_default_options()

        d, X, stats = MS.execute(options, K, M, lambda_c=1e-8, tol=1e-6)
        eta = EA.compute_backward_error(K, M, d, X)

        self.assertTrue( NP.all(eta < 2*NP.finfo(dtype).eps) )



if __name__ == '__main__':
    unittest.main()
