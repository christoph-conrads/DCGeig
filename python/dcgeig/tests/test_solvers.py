#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2015-2016 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

import unittest

import ctypes
import numpy as NP
import numpy.linalg as NL
import numpy.matlib as M
import numpy.random as R

import dcgeig.solvers as gep_solvers
from dcgeig.error_analysis import compute_backward_error
import dcgeig.utils as U


def check_return_values(self, dtype, d, X):
    self.assertEqual( type(X), M.matrix )
    self.assertEqual( X.dtype, dtype )

    self.assertEqual( type(d), NP.ndarray )
    self.assertEqual( d.dtype, dtype )
    self.assertEqual( d.size, X.shape[1] )



def rotate_matrix(A):
    assert U.is_hermitian(A)

    dtype = A.dtype
    n = len(A)

    r = R.RandomState(seed=0)
    B = r.uniform(low=-1, high=+1, size=[n,n])
    Q, _ = NL.qr(B)
    Q = M.matrix( Q.astype(dtype) )

    QAQH = U.force_hermiticity( Q * A * Q.H )
    return QAQH



def test_simple(self, f):
    n = 4;

    for dtype in self.dtypes:
        I = M.eye(n, dtype=dtype)
        d, X = f(I, I)

        check_return_values(self, dtype, d, X)

        eta = compute_backward_error(I, I, d, X)

        epsilon = NP.finfo(dtype).eps
        self.assertTrue( NP.all(eta <= n * epsilon) )



def test_random(self, f, reltol=1):
    n = 8;

    for dtype in self.dtypes:
        r = R.RandomState(seed=0)

        for i in range(16):
            L_A = M.matrix( r.random_sample([n,n]), dtype=dtype )
            L_B = M.matrix( r.random_sample([n,n]), dtype=dtype )

            A = U.force_hermiticity(L_A * L_A.H)
            B = U.force_hermiticity(L_B * L_B.H)

            d, X = f(A, B)

            check_return_values(self, dtype, d, X)

            eta = compute_backward_error(A, B, d, X)

            epsilon = NP.finfo(dtype).eps
            self.assertTrue( NP.all(eta <= reltol * n * epsilon) )



def test_random_singular(self, f, reltol=1):
    n = 8;

    for dtype in self.dtypes:
        r = R.RandomState(seed=0)

        for i in range(16):
            L_A = M.matrix( NP.tril(r.random_sample([n,n])), dtype=dtype )
            L_B = M.matrix( NP.tril(r.random_sample([n,n])), dtype=dtype )

            L_A[0,0] = 0
            L_B[-1,-1] = 0

            A = U.force_hermiticity(L_A * L_A.H)
            B = U.force_hermiticity(L_B * L_B.H)

            d, X = f(A, B)

            check_return_values(self, dtype, d, X)

            eta = compute_backward_error(A, B, d, X)

            epsilon = NP.finfo(dtype).eps
            self.assertTrue( NP.all(eta <= reltol * n * epsilon) )




def test_checks(self, f):
    n = 4

    with self.assertRaises(ValueError):
        f( M.eye(n, dtype=NP.float32), M.eye(n, dtype=NP.float64) )

    with self.assertRaises(ValueError):
        f( M.eye(n, dtype=NP.float64), M.eye(n, dtype=NP.float32) )


    for dtype in self.dtypes:
        I = M.eye(n, dtype=dtype)

        with self.assertRaises(ValueError):
            Z = M.eye(0, dtype=dtype)
            d, X = f(Z, Z)


        with self.assertRaises(TypeError):
            I = M.eye(n, dtype=dtype)
            A = NP.eye(n, dtype=dtype)
            d, X = f(I, A)

        with self.assertRaises(TypeError):
            I = M.eye(n, dtype=dtype)
            A = NP.eye(n, dtype=dtype)
            d, X = f(A, I)


        with self.assertRaises(ValueError):
            J = M.eye(2*n, dtype=dtype)
            d, X = f(I, J)

        with self.assertRaises(ValueError):
            J = M.eye(2*n, dtype=dtype)
            d, X = f(J, I)


        with self.assertRaises(ValueError):
            A = M.matrix(range(n*n)).reshape([n,n])
            d, X = f(A, I)

        with self.assertRaises(ValueError):
            A = M.matrix(range(n*n)).reshape([n,n])
            d, X = f(I, A)



def test_small_dimensions(self, f):
    for dtype in self.dtypes:
        for n in [1,2,3]:
            A = dtype(NP.pi) * M.eye(n, dtype=dtype)
            B = dtype(NP.e)  * M.eye(n, dtype=dtype)

            d, X = f(A, B)

            check_return_values(self, dtype, d, X)

            eta = compute_backward_error(A, B, d, X)

            epsilon = NP.finfo(dtype).eps
            self.assertTrue( NP.all(eta <= 2*n * epsilon) )



def test_singular_matrix_2by2(self, f):
    n = 2

    for dtype in self.dtypes:
        epsilon = NP.finfo(dtype).eps

        F = rotate_matrix( M.eye(n, dtype=dtype) )
        S = rotate_matrix( M.matrix(NP.diag([1,0]), dtype=dtype) )

        d, X = f(F, S)
        check_return_values(self, dtype, d, X)
        eta = compute_backward_error(F, S, d, X)
        self.assertTrue( NP.all(eta <= M.sqrt(2) * n * epsilon) )

        d, X = f(S, F)
        check_return_values(self, dtype, d, X)
        eta = compute_backward_error(S, F, d, X)
        self.assertTrue( NP.all(eta <= M.sqrt(2) * n * epsilon) )



def test_singular_matrix_3by3(self, f):
    n = 3

    for dtype in self.dtypes:
        epsilon = NP.finfo(dtype).eps

        A = M.eye(n, dtype=dtype)
        B = M.eye(n, dtype=dtype)

        A[0,0] = 0
        B[1,1] = 0
        A[2,2] = 0
        B[2,2] = 0

        A = rotate_matrix(A)
        B = rotate_matrix(B)

        d, X = f(A, B)
        check_return_values(self, dtype, d, X)

        eta = compute_backward_error(A, B, d, X)
        self.assertTrue( NP.all(eta <= M.sqrt(2) * n * epsilon) )


def test_zero_matrix(self, f):
    n = 4;

    for dtype in self.dtypes:
        epsilon = NP.finfo(dtype).eps

        A = M.eye(n, dtype=dtype)
        B = M.zeros([n,n], dtype=dtype)

        d, X = f(A, B)
        check_return_values(self, dtype, d, X)
        eta = compute_backward_error(A, B, d, X)
        self.assertTrue( NP.all(eta <= n * epsilon) )

        d, X = f(B, A)
        check_return_values(self, dtype, d, X)
        eta = compute_backward_error(B, A, d, X)
        self.assertTrue( NP.all(eta <= n * epsilon) )



class Test_solve_with_qr_csd(unittest.TestCase):
    dtypes = [NP.float32, NP.float64]
    solver = gep_solvers.qr_csd

    def setUp(self):
        self.numpy_warnings = NP.seterr(invalid='ignore')

    def tearDown(self):
        NP.seterr(**self.numpy_warnings)



    def test_simple(self):
        test_simple(self, self.solver)

    def test_random(self):
        test_random(self, self.solver, NP.sqrt(2))

    def test_random_singular(self):
        test_random_singular(self, self.solver, 2)

    def test_checks(self):
        test_checks(self, self.solver)

    def test_small_dimensions(self):
        test_small_dimensions(self, self.solver)

    def test_singular_matrix_2by2(self):
        test_singular_matrix_2by2(self, self.solver)

    def test_singular_matrix_3by3(self):
        test_singular_matrix_3by3(self, self.solver)

    def test_zero_matrix(self):
        test_zero_matrix(self, self.solver)

    def test_workspace_query(self):
        for n in [1,2,3,10,100,1000,10000]:
            minimum = 2*n*n + max(18, 17*n - 4)
            opt = gep_solvers.qr_csd_workspace_query(n)

            self.assertTrue( opt > 0 )
            self.assertTrue( opt >= minimum )






class Test_solve_with_gsvd(unittest.TestCase):
    dtypes = [NP.float32, NP.float64]
    solver = gep_solvers.gsvd

    def setUp(self):
        self.numpy_warnings = NP.seterr(invalid='ignore')

    def tearDown(self):
        NP.seterr(**self.numpy_warnings)



    def test_simple(self):
        test_simple(self, self.solver)

    def test_random(self):
        test_random(self, self.solver)

    def test_random_singular(self):
        test_random_singular(self, self.solver)

    def test_checks(self):
        test_checks(self, self.solver)

    def test_small_dimensions(self):
        test_small_dimensions(self, self.solver)

    def test_singular_matrix_2by2(self):
        test_singular_matrix_2by2(self, self.solver)

    def test_singular_matrix_3by3(self):
        test_singular_matrix_3by3(self, self.solver)

    def test_zero_matrix(self):
        test_zero_matrix(self, self.solver)

    def test_workspace_query(self):
        for n in [1,2,3,10,100,1000,10000]:
            minimum = 3*n
            opt = gep_solvers.gsvd_workspace_query(n)

            self.assertTrue( opt > 0 )
            self.assertTrue( opt >= minimum )



class Test_deflate_gep(unittest.TestCase):
    dtypes = [NP.float32, NP.float64]


    def check_return_values(self, n, dtype, A, B, X, Q):
        self.assertEqual( type(A), M.matrix )
        self.assertEqual( A.shape[0], A.shape[1] );
        self.assertTrue( U.is_hermitian(A) )

        self.assertEqual( type(B), M.matrix )
        self.assertEqual( B.shape[0], B.shape[1] );
        self.assertTrue( U.is_hermitian(B) )

        self.assertEqual( type(X), M.matrix )
        self.assertEqual( X.shape[0], X.shape[1] );

        self.assertEqual( type(Q), M.matrix )
        self.assertEqual( Q.shape[0], Q.shape[1] );

        self.assertEqual( A.shape[0], B.shape[0] )
        self.assertTrue( A.shape[0] >= 0 )
        self.assertTrue( A.shape[0] <= n )

        self.assertEqual( X.shape[0], n )
        self.assertEqual( Q.shape[0], n )



    def test_mass_matrix_full_rank(self):
        n = 4

        for dtype in self.dtypes:
            A = M.eye( n, dtype=dtype )
            B = M.eye( n, dtype=dtype )

            A0, B0, X, Q = gep_solvers.deflate_gep(A, B)
            self.check_return_values(n, dtype, A0, B0, X, Q)

            self.assertEqual( A0.shape[0], n )
            self.assertEqual( B0.shape[0], n )

            self.assertTrue( NP.all(A0 == A) )
            self.assertTrue( NP.all(B0 == B) )



    def test_mass_matrix_singular(self):
        n = 4

        for dtype in self.dtypes:
            A = M.eye( n, dtype=dtype )
            B = M.eye( n, dtype=dtype )
            B[0,0] = 0

            A0, B0, X, Q = gep_solvers.deflate_gep(A, B)
            self.check_return_values(n, dtype, A0, B0, X, Q)

            self.assertEqual( A0.shape[0], n-1 )
            self.assertEqual( B0.shape[0], n-1 )



    def test_error_detection(self):
        n = 4

        for dtype in self.dtypes:
            # mass matrix indefinite
            with self.assertRaises(ValueError):
                A = M.eye( n, dtype=dtype )
                B = M.eye( n, dtype=dtype )
                B[0,0] = -1

                A0, B0, X, Q = gep_solvers.deflate_gep(A, B)


            # matrix pencil nonregular
            with self.assertRaises(ValueError):
                A = M.eye( n, dtype=dtype )
                B = M.eye( n, dtype=dtype )
                A[0,0] = 0
                B[0,0] = 0

                A0, B0, X, Q = gep_solvers.deflate_gep(A, B)



    def test_workspace_query(self):
        for n in [2,3,10,100,1000,10000]:
            minimum = 2*n*n + 6*n + 1
            lwork, liwork = gep_solvers.deflate_gep_workspace_query(n)

            self.assertTrue( lwork > 0 )
            self.assertTrue( lwork >= minimum )

            self.assertTrue( liwork >= 5*n + 3 )



class Test_solve_with_deflation(unittest.TestCase):
    dtypes = [NP.float32, NP.float64]
    solver = gep_solvers.deflation

    def setUp(self):
        self.numpy_warnings = NP.seterr(invalid='ignore')

    def tearDown(self):
        NP.seterr(**self.numpy_warnings)



    def test_simple(self):
        test_simple(self, self.solver)

    def test_random(self):
        test_random(self, self.solver, 10)

    def test_checks(self):
        test_checks(self, self.solver)

    def test_small_dimensions(self):
        test_small_dimensions(self, self.solver)

    def test_singular_matrix_2by2(self):
        test_singular_matrix_2by2(self, self.solver)

    def test_zero_matrix(self):
        test_zero_matrix(self, self.solver)

    def test_workspace_query(self):
        for n in [2,3,10,100,1000,10000]:
            lwork_min = 4*n*n + 6*n + 1
            liwork_min = 5*n + 3
            lwork, liwork = gep_solvers.deflation_workspace_query(n)

            self.assertTrue( lwork > 0 )
            self.assertTrue( lwork >= lwork_min )

            self.assertTrue( liwork > 0 )
            self.assertTrue( liwork >= liwork_min )



if __name__ == '__main__':
    unittest.main()
