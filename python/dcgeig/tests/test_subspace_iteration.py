#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import unittest

import dcgeig.subspace_iteration as SI

import numpy as NP
import numpy.matlib as ML
import numpy.random

import scipy.linalg as SL
import scipy.sparse as SS



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

        x = SI.inverse_iteration(solve, K, M, b, 10*min(ds))
        x = x / SL.norm(x)

        self.assertTrue( X[:,0].H * x > tol )
        self.assertEqual( x.dtype, b.dtype )

        y = ML.copy(b)
        ret = SI.inverse_iteration(solve, K, M, y, 10*min(ds), overwrite_b=True)
        y = y / SL.norm(y)

        self.assertTrue( X[:,0].H * y > tol )
        self.assertEqual( y.dtype, b.dtype )
        self.assertEqual( SL.norm(x-y), 0 )
