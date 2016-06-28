#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2015-2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import unittest

import numpy as N
import numpy.matlib as M
import dcgeig.utils as U



class Test_force_hermiticity(unittest.TestCase):
    def test_empty(self):
        self.assertTrue( U.is_hermitian(U.force_hermiticity(M.eye(0))) )


    def test_random(self):
        n = 8

        A = M.rand(n,n)
        self.assertFalse( U.is_hermitian(A) )

        B = U.force_hermiticity(A)
        self.assertTrue( U.is_hermitian(B) )



class Test_is_hermitian(unittest.TestCase):
    def test_empty(self):
        self.assertTrue( U.is_hermitian(M.eye(0)) )
        self.assertTrue( U.is_hermitian(M.eye(0)+0j) )



    def test_simple(self):
        n = 8

        dtypes = [N.float32, N.float64, N.complex64, N.complex128]

        for dtype in dtypes:
            self.assertTrue( U.is_hermitian(M.eye  (n,     dtype=dtype)) )
            self.assertTrue( U.is_hermitian(M.zeros([n,n], dtype=dtype)) )
            self.assertFalse( \
                U.is_hermitian(M.matrix([[0,1],[2,3]],dtype=dtype)) )

            x = N.array( [1.0, 2.0, -3.0, 4.0], dtype=dtype )
            self.assertFalse( U.is_hermitian(N.matrix(N.vander(x))) )



if __name__ == '__main__':
    unittest.main()
