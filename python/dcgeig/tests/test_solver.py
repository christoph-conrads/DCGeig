#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import unittest

import numpy as NP
import numpy.matlib as ML

import scipy.sparse as SS

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

        self.assertTrue( abs(d_max - 3) <= tol )


    # arpack does not work with 1x1 matrices
    def test_1by1(self):
        n = 3
        K = SS.spdiags(1.0*NP.arange(1,n+1), 0, n, n)
        M = SS.identity(n, dtype=K.dtype)
        v = ML.eye(n, 1)

        d = solver.compute_largest_eigenvalue(K, M, v)

        self.assertEqual(d, 1)



if __name__ == '__main__':
    unittest.main()
