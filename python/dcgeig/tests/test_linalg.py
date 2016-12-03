#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

import numbers

import unittest

import dcgeig.error_analysis as error_analysis
import dcgeig.linalg as linalg

import numpy as NP
import numpy.linalg as NL
import numpy.matlib as ML

import scipy.sparse as SS



class Test_spll(unittest.TestCase):
    def test_simple(self):
        n = 5
        A = SS.diags( [1.0 * NP.arange(1, n+1)], [0], [n,n], format='csc' )

        LL = linalg.spll(A)

        b = 2*3*4*5 * NP.ones([n,1])
        x = LL.solve(b)

        self.assertEqual( NL.norm(A*x - b), 0 )



class Test_orthogonalize(unittest.TestCase):
    def test_simple(self):
        for dtype in [NP.float32, NP.float64]:
            raw = [[2, 0, 0], [0, 2, 1], [0, 1, 2], [0, 0, 0]]
            U = ML.matrix(raw, dtype=dtype)
            n = U.shape[1]

            Q = linalg.orthogonalize(U)

            self.assertEqual( Q.shape[0], U.shape[0] )
            self.assertEqual( Q.shape[1], U.shape[1] )

            eps = NP.finfo(dtype).eps
            self.assertEqual( Q[0,0], 1 )
            self.assertTrue( NL.norm(Q.H * Q - NP.eye(n)) <= n*eps )
            self.assertEqual( NL.norm(Q[-1,:]), 0 )



class Test_rayleigh_ritz(unittest.TestCase):
    def test_simple(self):
        n = 5
        m = 2

        ds = NP.arange(1.0 * n)
        K = SS.spdiags(ds, 0, n, n, format='csc')
        M = SS.identity(n, format='lil')

        S = ML.zeros( [n,m] )
        S[2:4,:] = NP.array( [[1, 1], [1, 0]] )

        d, X = linalg.rayleigh_ritz(K, M, S)
        eta, delta = error_analysis.compute_errors(K, M, d, X)

        eps = NP.finfo(d.dtype).eps
        self.assertTrue( NP.all(eta <= n * eps) )



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

        d_max = linalg.compute_largest_eigenvalue(K, M, S, tol=tol)

        self.assertIsInstance(d_max, numbers.Real)
        self.assertTrue( abs(d_max - 3) <= tol )


    # arpack does not work with 1x1 matrices
    def test_1by1(self):
        n = 3
        K = SS.spdiags(1.0*NP.arange(1,n+1), 0, n, n)
        M = SS.identity(n, dtype=K.dtype)
        v = ML.eye(n, 1)

        d = linalg.compute_largest_eigenvalue(K, M, v)

        self.assertIsInstance(d, numbers.Real)
        self.assertEqual(d, 1)



if __name__ == '__main__':
    unittest.main()
