#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import unittest

import dcgeig.error_analysis as error_analysis
import dcgeig.gallery as gallery
import dcgeig.subspace_iteration as subspace_iteration

import numpy as NP
import numpy.matlib as ML
import numpy.random

import scipy.linalg as SL
import scipy.sparse as SS
import scipy.sparse.linalg as LA



class Test_inverse_iteration(unittest.TestCase):
    def test_simple(self):
        n = 4
        dtype = NP.float32

        random = numpy.random.RandomState(seed=1)
        X, _ = SL.qr( 2*random.rand(n,n) - 1 )
        X = ML.matrix(X, dtype=dtype)

        ds = NP.concatenate( [[0.05], NP.arange(1, n)] ).astype(dtype)
        D = ML.matrix( NP.diag(ds) )

        K = SS.csc_matrix( X * D * X.H )
        M = SS.identity(n, dtype=dtype)
        solve = lambda B: SL.solve(K.todense(), B)

        b = NP.float32(1e-4) * X[:,0] + X[:,3:] * NP.ones([n-3,1])
        b = b.astype(dtype)
        d = NP.array(b.H * K * b).reshape( (1,) )

        tol = 0.99

        x = subspace_iteration.inverse_iteration(solve, K, M, b, 10*min(ds))
        x = x / SL.norm(x)

        self.assertTrue( X[:,0].H * x > tol )
        self.assertEqual( x.dtype, b.dtype )

        y = ML.copy(b)
        ret = subspace_iteration.inverse_iteration( \
                solve, K, M, y, 10*min(ds), overwrite_b=True)
        y = y / SL.norm(y)

        self.assertTrue( X[:,0].H * y > tol )
        self.assertEqual( y.dtype, b.dtype )
        self.assertEqual( SL.norm(x-y), 0 )



class Test_subspace_iteration(unittest.TestCase):
    def test_no_eigenpairs_below_cutoff(self):
        n = 4
        K = SS.identity(n)
        M = SS.identity(n)
        solve = lambda x: x

        lambda_c = 0.5
        x0 = ML.matrix(1.0 * NP.arange(1, n+1)).H

        d, X, eta, delta = \
            subspace_iteration.execute(solve, K, M, x0, lambda_c, 1e-8, 1e-2)

        self.assertEqual( d.size, 1 )
        self.assertTrue( min(d-delta) > lambda_c )


    def test_FDM_Laplacian_1D(self):
        n = 5
        K = gallery.fdm_laplacian_1D(n)
        M = SS.identity(n)
        solve = lambda b: LA.spsolve(K, b).reshape(b.shape)

        # test with a subspace of dimension 2, one desired eigenpair,
        # eigenvector for smallest eigenvalue in search space
        xs = NP.sin(NP.pi * NP.arange(1,n+1) / (n+1))
        b = ML.matrix( [xs, NP.ones(n)] ).H

        d_min = 2 * NP.pi**2

        d, X, _, delta = \
            subspace_iteration.execute(solve, K, M, b, 1.5*d_min, 1e-8, 1e-2)

        t = d - delta <= 1.5*d_min
        self.assertEqual( NP.sum(t), 1 )

        self.assertEqual( X.shape[0], K.shape[0] )
        self.assertEqual( X.shape[1], d.size )

        eta, delta = error_analysis.compute_errors(K, M, d[t], X[:,t])

        self.assertTrue( eta < 1e-15 )
        self.assertTrue( delta < 1e-13 )



if __name__ == '__main__':
    unittest.main()
