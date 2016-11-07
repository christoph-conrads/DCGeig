#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import unittest

import dcgeig.gallery as gallery
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


    def test_overwrite(self):
        for dtype in [NP.float32, NP.float64]:
            raw = [[2, 0, 0], [0, 2, 1], [0, 1, 2], [0, 0, 0]]
            U = ML.matrix(raw, dtype=dtype)
            n = U.shape[1]

            Q = U.copy()
            linalg.orthogonalize(Q, do_overwrite=True)

            self.assertEqual( Q.shape[0], U.shape[0] )
            self.assertEqual( Q.shape[1], U.shape[1] )

            eps = NP.finfo(dtype).eps
            self.assertEqual( Q[0,0], 1 )
            self.assertTrue( NL.norm(Q.H * Q - NP.eye(n)) <= n*eps )
            self.assertEqual( NL.norm(Q[-1,:]), 0 )



if __name__ == '__main__':
    unittest.main()
