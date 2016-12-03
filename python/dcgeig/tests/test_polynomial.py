#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

import unittest

import numpy as NP
import numpy.linalg as LA
import numpy.polynomial.chebyshev as NPC

import scipy.sparse as SS
import scipy.sparse.linalg as SLA

import dcgeig.gallery as gallery
import dcgeig.polynomial as polynomial



class Test_compute_jackson_coefficients(unittest.TestCase):
    def test_degree_0(self):
        js = polynomial.compute_jackson_coefficients(0)

        self.assertIsInstance( js, NP.ndarray )
        self.assertEqual( js.size, 1 )
        self.assertEqual( js[0], 1 )


    def test_degrees(self):
        for d in range(1,10):
            js = polynomial.compute_jackson_coefficients(d)

            self.assertIsInstance( js, NP.ndarray )

            self.assertEqual( js.size, d+1 )
            self.assertEqual( js[0], 1 )
            self.assertEqual( js[-1], 0 )

            self.assertTrue( NP.all(js[:-1] > 0) )
            self.assertTrue( NP.all(js[1:] < 1) )



class Test_compute_chebyshev_heaviside_coefficients(unittest.TestCase):
    def test_simple(self):
        for d in range(0, 10):
            cs = polynomial.compute_chebyshev_heaviside_coefficients(d)

            self.assertIsInstance( cs, NP.ndarray )
            self.assertEqual( cs.size, d+1 )
            self.assertEqual( cs[0], 0.5 )
            self.assertTrue( LA.norm(cs) < 1 )



class Test_compute_chebyshev_step_coefficients(unittest.TestCase):
    def test_simple(self):
        for d in range(0, 10):
            for a in NP.arange(-1, +1, 0.1):
                for b in NP.arange(a+0.1, +1, 0.1):
                    cs = polynomial.compute_chebyshev_step_coefficients(d, a, b)

                    self.assertIsInstance( cs, NP.ndarray )
                    self.assertEqual( cs.size, d+1 )
                    self.assertTrue( min(cs) > -1 )
                    self.assertTrue( max(cs) < +1 )



class Test_evaluate_matrix_polynomial(unittest.TestCase):
    def test_simple(self):
        for n in range(1, 5):
            ps = NP.full(1, 1.0)
            solve = lambda x: x
            K = SS.identity(n)
            M = SS.identity(n)

            u = NP.ones([n,1])
            v = polynomial.evaluate_matrix_polynomial(ps, 1, 0, solve, K, M, u)

            self.assertEqual( LA.norm(u-v), 0 )


    def test_max_exponent(self):
        n = 4

        for d in range(1, 5):
            ps = NP.full(d+1, 0.0)
            ps[-1] = 1
            solve = lambda x: x
            K = SS.identity(n)
            M = SS.identity(n)

            u = NP.reshape(NP.arange(1, n+1), [n,1])
            v = polynomial.evaluate_matrix_polynomial(ps, 2, 0, solve, K, M, u)

            self.assertTrue( LA.norm(2**d * v - u), 0 )


    def test_all_ones_monomial(self):
        n = 4
        m = 3

        for d in range(1, 5):
            ps = NP.ones(d+1)
            cs = NPC.poly2cheb(ps)
            solve = lambda x: x
            K = SS.identity(n)
            M = SS.identity(n)

            U = 1.0 * NP.reshape(NP.arange(1, m*n+1), [n,m])
            V = polynomial.evaluate_matrix_polynomial(cs, 1, 0, solve, K, M, U)

            self.assertEqual( LA.norm(V - (d+1)*U), 0 )



class Test_approximate_projection(unittest.TestCase):
    def test_simple(self):
        n = 4

        l = 0.75
        r = 1.5

        K = SS.diags( 1.0 * NP.arange(1,n+1), format='csc' )
        M = SS.identity(n)

        for d in [25,50,75]:
            f = polynomial.approximate_projection(d, l, r, K, M)

            U = NP.identity(n)
            V = f(U)

            n_c = NP.trace(V)
            self.assertTrue( abs(n_c - 1) <= 2e-2 )


    # test tridiagonal stiffness matrix + identity
    def test_1D_FDM_Laplacian(self):
        n = 100
        A = gallery.fdm_laplacian_1D(n)
        B = SS.identity(n)

        l = NP.pi**2
        r = 2 * l

        for d in [25,50,75]:
            f = polynomial.approximate_projection(d, l, r, A, B)

            U = NP.identity(n)
            V = f(U)

            n_c = NP.trace(V)
            self.assertTrue( abs(n_c - 1) <= 2e-2 )


    # test block tridiagonal stiffness matrix + identity
    def test_2D_FDM_Laplacian(self):
        n = 10
        A = gallery.fdm_laplacian_2D(n)
        B = SS.identity(n*n)

        l = 1.5 * NP.pi**2
        r = 2 * l

        for d in [25,50,75]:
            f = polynomial.approximate_projection(d, l, r, A, B)

            U = NP.identity(n**2)
            V = f(U)

            n_c = NP.trace(V)
            self.assertTrue( abs(n_c - 1) <= .15 )


    # test tridiagonal + tridiagonal matrix
    def test_1D_FEM_Laplacian(self):
        n = 100
        A, B = gallery.fem_laplacian_1D(n)

        l = 0.75 * NP.pi**2
        r = 2 * l

        for d in [25,50,75]:
            f = polynomial.approximate_projection(d, l, r, A, B)

            U = NP.identity(n)
            V = f(U)

            n_c = NP.trace(V)
            self.assertTrue( abs(n_c - 1) <= 2e-2 )


    # test block tridiagonal + block tridiagonal matrix
    def test_2D_FEM_Laplacian(self):
        n = 10
        A, B = gallery.fem_laplacian_2D(n)

        l = 1.5 * NP.pi**2
        r = 2 * l

        for d in [25,50,75]:
            f = polynomial.approximate_projection(d, l, r, A, B)

            U = NP.identity(n**2)
            V = f(U)

            n_c = NP.trace(V)
            self.assertTrue( abs(n_c - 1) <= 1e-1 )



if __name__ == '__main__':
    unittest.main()
