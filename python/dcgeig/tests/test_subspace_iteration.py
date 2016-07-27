#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import unittest

import dcgeig.subspace_iteration as SI
import dcgeig.utils as utils

import numpy as NP
import numpy.matlib as ML
import numpy.random

import scipy.linalg as SL



class Test_chebychev(unittest.TestCase):
    def test_simple(self):
        dtype = NP.float32
        n = 25

        random = numpy.random.RandomState(seed=1)

        ds = NP.concatenate( [[0.05], NP.arange(1, n)] ).astype(dtype)
        D = ML.matrix( NP.diag(ds) )

        X, _ = SL.qr( 2*random.rand(n,n) - 1 )
        X = ML.matrix(X, dtype=dtype)

        K = utils.force_hermiticity( X * D * X.H )
        M = ML.identity(n, dtype=dtype)
        solve = lambda B: SL.solve(K, B)

        u = 1e-3 * X[:,0] + X[:,3:] * NP.ones([n-3,1])
        u = u.astype(dtype)

        c = (1 + 1/max(ds)) / 2.0
        e = (1 - 1/max(ds)) / 2.0
        v = SI.chebychev(3, c, e, solve, K, M, u)
        v = v / SL.norm(v)

        self.assertTrue( abs(X[:,0].H * v) > 0.9 )
        self.assertEqual( v.dtype, u.dtype )



class Test_inverse_iteration(unittest.TestCase):
    def test_simple(self):
        n = 4
        dtype = NP.float32

        lambda_c = 1.0
        degree = 2

        random = numpy.random.RandomState(seed=1)

        ds = NP.concatenate( [[0.05], NP.arange(1, n)] ).astype(dtype)
        D = ML.matrix( NP.diag(ds) )

        X, _ = SL.qr( 2*random.rand(n,n) - 1 )
        X = ML.matrix(X, dtype=dtype)

        K = utils.force_hermiticity( X * D * X.H )
        M = ML.identity(n, dtype=dtype)
        solve = lambda B: SL.solve(K, B)

        b = NP.float32(1e-3) * X[:,0] + X[:,3:] * NP.ones([n-3,1])
        b = b.astype(dtype)
        d = NP.array(b.H * K * b).reshape( (1,) )
        eta = 0.1
        delta = 0.1*d

        x = SI.inverse_iteration(lambda_c, degree, solve, K, M, d, b, eta,delta)
        x = x / SL.norm(x)

        self.assertTrue( X[:,0].H * x > 0.9 )
        self.assertEqual( x.dtype, b.dtype )

        y = ML.copy(b)
        ret = SI.inverse_iteration( \
                lambda_c, degree, solve, K, M, d, y,eta,delta, overwrite_b=True)
        y = y / SL.norm(y)

        self.assertTrue( X[:,0].H * y > 0.9 )
        self.assertEqual( y.dtype, b.dtype )
        self.assertEqual( SL.norm(x-y), 0 )
