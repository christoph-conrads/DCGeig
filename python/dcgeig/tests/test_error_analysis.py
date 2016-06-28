#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2015-2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import unittest

import numpy as NP
import numpy.matlib as ML

import dcgeig.error_analysis as EA



class Test_compute_backward_error(unittest.TestCase):
    def test_simple(self):
        n = 2;

        I = ML.eye(n)
        d = NP.ones(n)
        epsilon = NP.finfo(NP.float64).eps

        eta = EA.compute_backward_error(I, I, d, I)

        self.assertEqual( type(eta), NP.ndarray )
        self.assertEqual( eta.dtype, NP.float64 )
        self.assertTrue( (eta >= 0).all() )
        self.assertTrue( (eta < epsilon).all() )


    def test_bounded_by_one(self):
        n = 1;

        I = ML.eye(n)
        d = NP.zeros(n)
        epsilon = NP.finfo(NP.float64).eps

        eta = EA.compute_backward_error(I, I, d, I)

        self.assertEqual( type(eta), NP.ndarray )
        self.assertEqual( eta.dtype, NP.float64 )
        self.assertTrue( (eta >= 0).all() )
        self.assertTrue( (eta <= 1).all() )


    def test_empty(self):
        I = ML.eye(0)
        d = NP.zeros(0)
        eta = EA.compute_backward_error(I, I, d, I)

        self.assertEqual( type(eta), NP.ndarray )
        self.assertEqual( eta.dtype, NP.float64 )
        self.assertEqual( eta.ndim, 1 )
        self.assertEqual( eta.shape, (0,) )



    def test_infinite(self):
        n = 4
        I = ML.eye(n)
        x = ML.ones( [n,1] ) / NP.sqrt(n)
        d = NP.array([NP.inf])

        eta = EA.compute_backward_error(I, I, d, x)

        self.assertEqual( type(eta), NP.ndarray )
        self.assertEqual( eta.dtype, NP.float64 )
        self.assertTrue( eta >= 0 )
        self.assertFalse( NP.isinf(eta) )


    def test_eigenvector_normalization(self):
        n = 4
        I = ML.eye(n)
        X = 10 * ML.matrix( NP.diag(range(1,n+1)).astype(NP.float64) )
        d = NP.ones(n)
        epsilon = NP.finfo(NP.float64).eps

        eta = EA.compute_backward_error(I, I, d, X)

        self.assertEqual( type(eta), NP.ndarray )
        self.assertEqual( eta.dtype, NP.float64 )
        self.assertTrue( NP.all(eta >= 0) )
        self.assertTrue( NP.all(eta <= n * epsilon) )


    def test_precondition_checks(self):
        n = 4

        I = ML.eye(n) + 0j
        X = ML.ones( [n,2] ) + 1.0j
        d = NP.array( [1.0, 2.0] ) + 0j

        with self.assertRaises(ValueError):
            eta = EA.compute_backward_error(I+1j, I, d, X)

        with self.assertRaises(ValueError):
            eta = EA.compute_backward_error(I, I, d+1j, X)


    def test_nullspace_intersection(self):
        n = 3;

        A = ML.eye(n)
        B = ML.eye(n)

        A[0,0] = 0
        B[1,1] = 0
        A[2,2] = 0
        B[2,2] = 0

        X = ML.eye(n)
        d = NP.array( [0, NP.inf, -1] )

        eta = EA.compute_backward_error(A, B, d, X)
        self.assertTrue( (eta == 0).all() )


    def test_zero(self):
        n = 3;
        Z = ML.zeros([n,n])
        X = ML.eye(n)
        d = NP.array( [0, 1, NP.inf] );

        eta = EA.compute_backward_error(Z, Z, d, X)
        self.assertTrue( (eta == 0).all() )


    def test_complex(self):
        n = 3;
        dtype = NP.complex128
        K = 10 * ML.eye(n, dtype=dtype)
        K[0,1] = 0.0+1.0j
        K[1,0] = 0.0-1.0j

        M = ML.eye(n, dtype=dtype)
        d = NP.full(1, 1.0, dtype=dtype)
        x = ML.matrix( NP.arange(1,n+1,dtype=dtype).reshape([n,1]) )

        eta = EA.compute_backward_error(K, M, d, x)

        self.assertTrue( NP.isrealobj(eta) )
        self.assertTrue( eta >= 0 )



class Test_compute_condition_number(unittest.TestCase):
    def setUp(self):
        self.numpy_settings = NP.seterr(divide='ignore')

    def tearDown(self):
        NP.seterr(divide=self.numpy_settings['divide'])



    def test_simple(self):
        n = 2;

        I = ML.eye(n)
        d = NP.ones(n)
        epsilon = NP.finfo(NP.float64).eps

        kappa = EA.compute_condition_number(I, I, d, I)
        self.assertTrue( (kappa >= 0).all() )
        self.assertTrue( (kappa <= n).all() )

        J = ML.eye(n) + 0j
        kappa = EA.compute_condition_number(J, J, d+0j, J)
        self.assertTrue( (kappa >= 0).all() )
        self.assertTrue( (kappa <= n).all() )



    def test_infinite(self):
        n = 2;

        I = ML.eye(n)
        A = ML.eye(n)
        B = ML.zeros( [n,n] )
        d = NP.ones(n)

        kappa = EA.compute_condition_number(A, B, d, I)
        self.assertTrue( NP.all(NP.isinf(kappa)) )

        kappa = EA.compute_condition_number(A+0j, B+0j, d+0j, I+0j)
        self.assertTrue( NP.all(NP.isinf(kappa)) )



    def test_preconditions(self):
        n = 8;

        I = ML.eye(n)
        J = ML.eye(n) + 0j
        d = NP.ones(n)

        with self.assertRaises(ValueError):
            # use complex identity matrices to prevent dtype mismatches
            kappa = EA.compute_condition_number(J, J, 1j*d, J)

        with self.assertRaises(ValueError):
            kappa = EA.compute_condition_number(I, I, d / NP.float64(0), I)

        with self.assertRaises(ValueError):
            kappa = EA.compute_condition_number(I, I, -1*d, I)



    def test_empty(self):
        I = ML.eye(0)
        d = ML.zeros(0)

        kappa = EA.compute_condition_number(I, I, d, I)
        self.assertEqual( type(kappa), NP.ndarray )
        self.assertEqual( kappa.dtype, NP.float64 )
        self.assertEqual( kappa.ndim, 1 )
        self.assertEqual( kappa.shape, (0,) )



if __name__ == '__main__':
    unittest.main()
