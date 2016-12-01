#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.


import unittest

import numpy as NP

import scipy.linalg as SL
import scipy.sparse as SS
import scipy.sparse.linalg as LA

from dcgeig import gallery



def check_properties_1D(test, n, l, dtype, format, A, a_ii, a_ij):
    test.assertEqual( A.shape[0], A.shape[1] )
    test.assertEqual( A.dtype, dtype )
    test.assertEqual( A.getformat(), format )

    test.assertEqual( (A-A.H).nnz, 0 )
    test.assertEqual( SS.triu(A, +2).nnz, 0 )

    if l == n+1:
        test.assertTrue( NP.all(A.diagonal() == a_ii) )
        i, j, v = SS.find(SS.triu(A,+1))

        test.assertTrue( NP.all(v == a_ij) )




class Test_fdm_laplacian_1D(unittest.TestCase):
    def test_simple(self):
        for l in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
            for n in range(1,5):
                for dtype in [NP.float32, NP.float64]:
                    for format in ['csc', 'csr']:
                        A = gallery.fdm_laplacian_1D(n, l, dtype=dtype, format=format)

                        check_properties_1D(self, n, l, dtype, format, A, 2, -1)



class Test_fdm_laplacian(unittest.TestCase):
    def execute_test_simple(self, ns, ls, dtype, format):
        A = gallery.fdm_laplacian(ns, ls, dtype=dtype, format=format)
        d, X = gallery.fdm_laplacian_eigenpairs(ns, ls, dtype=dtype)
        m = reduce(lambda a, b: a*b, ns, 1)

        self.assertEqual( A.shape[0], m )
        self.assertEqual( (A-A.H).nnz, 0 )
        self.assertEqual( A.dtype, dtype )
        self.assertEqual( A.getformat(), format )

        self.assertEqual( d.size, m )
        self.assertEqual( d.dtype, dtype )
        self.assertTrue( NP.all(d == NP.sort(d)) )

        self.assertEqual( X.shape[0], m )
        self.assertEqual( X.shape[1], m )
        self.assertEqual( X.dtype, dtype )

        R = A*X - NP.multiply(X, d)
        rs = SL.norm(R, axis=0)

        eps = NP.finfo(dtype).eps
        self.assertTrue( max(rs) <= m*eps*LA.norm(A) )

        # test if eigenvectors are orthonormal
        self.assertTrue( SL.norm(X.H * X - NP.identity(m)) <= 2*m*eps )


    def test_simple(self):
        ns = [1, 2, 3, 4, 5]
        ls = [0.5, 1.0, 1.5, 2.0, 3.0]

        for k in range(1, len(ns)+1):
            for dtype in [NP.float32, NP.float64]:
                for format in ['csc', 'csr']:
                    self.execute_test_simple(ns[:k], ls[:k], dtype, format)



class Test_fem_laplacian_1D(unittest.TestCase):
    def test_simple(self):
        for l in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
            for n in range(1,5):
                for dtype in [NP.float32, NP.float64]:
                    for format in ['csc', 'csr']:
                        K,M = gallery.fem_laplacian_1D(n, l, dtype=dtype, format=format)

                        check_properties_1D(self, n, l, dtype, format, K, 2, -1)
                        check_properties_1D(self, n, l, dtype, format,6*M,4, +1)



class Test_fem_laplacian(unittest.TestCase):
    def execute_test_simple(self, ns, ls, dtype, format):
        K, M = gallery.fem_laplacian(ns, ls, dtype=dtype, format=format)
        d, X = gallery.fem_laplacian_eigenpairs(ns, ls, dtype=dtype)
        m = reduce(lambda a, b: a*b, ns, 1)

        self.assertEqual( K.shape[0], m )
        self.assertEqual( (K-K.H).nnz, 0 )
        self.assertEqual( K.dtype, dtype )
        self.assertEqual( K.getformat(), format )

        self.assertEqual( M.shape[0], m )
        self.assertEqual( (M-M.H).nnz, 0 )
        self.assertEqual( M.dtype, dtype )
        self.assertEqual( M.getformat(), format )

        self.assertEqual( d.dtype, dtype)
        self.assertEqual( d.size, m )
        self.assertTrue( NP.all(d == NP.sort(d)) )

        self.assertEqual( X.dtype, dtype)
        self.assertEqual( X.shape[0], m )
        self.assertEqual( X.shape[1], m )

        # compute approximate backward error
        R = K*X - NP.multiply(M*X,d)
        nominator = SL.norm(R, axis=0)
        denominator = NP.sqrt(LA.norm(K)**2 + d**2*LA.norm(M)**2)
        eta = nominator/denominator

        eps = NP.finfo(dtype).eps
        self.assertTrue( max(eta) <= m*eps )

        # test if eigenvectors are orthonormal
        self.assertTrue( SL.norm(X.H * X - NP.identity(m)) <= 2*m*eps )

        # test simultaneous diagonalization
        A = X.H * K * X
        self.assertTrue( SL.norm(SL.triu(A, +1)) <= m*eps*LA.norm(K) )

        B = X.H * M * X
        self.assertTrue( SL.norm(SL.triu(B, +1)) <= m*eps*LA.norm(M) )


    def test_simple(self):
        ns = [1, 2, 3, 4, 5]
        ls = [0.5, 1.0, 1.5, 2.0, 3.0]

        for k in range(1, len(ns)+1):
            for dtype in [NP.float32, NP.float64]:
                for format in ['csc', 'csr']:
                    self.execute_test_simple(ns[:k], ls[:k], dtype, format)



if __name__ == '__main__':
    unittest.main()
