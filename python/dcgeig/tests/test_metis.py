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

import dcgeig.metis as metis
import numpy as NP
import scipy.sparse as SS



class Test_bisection(unittest.TestCase):
    def check_return_value(self, n, t):
        self.assertTrue( isinstance(t, NP.ndarray) )
        self.assertEqual( t.dtype, bool )
        self.assertEqual( len(t.shape), 1 )
        self.assertEqual( t.size, n )


    def test_simple(self):
        for n in [1,2,3,4]:
            A = SS.identity(n, dtype=NP.float32, format='csc')
            t = metis.bisection(A)
            self.check_return_value(n, t)


    def test_2by2_a(self):
        n = 4
        A = SS.identity(n, dtype=NP.float32, format='lil')
        A[0,1] = 1
        A[1,0] = 1
        t = metis.bisection(A)

        self.check_return_value(n, t)
        self.assertTrue( t[0] == t[1] )
        self.assertTrue( t[2] == t[3] )
        self.assertTrue( t[0] != t[2] )



    def test_2by2_b(self):
        n = 4
        A = SS.identity(n, dtype=NP.float32, format='lil')
        A[1,2] = 1
        A[2,1] = 1
        t = metis.bisection(A)

        self.check_return_value(n, t)
        self.assertTrue( t[0] == t[3] )
        self.assertTrue( t[1] == t[2] )
        self.assertTrue( t[0] != t[1] )



    def test_value_range(self):
        n = 4
        A = SS.identity(n, dtype=NP.float32, format='lil')

        A[1,0] = A[0,1] = -1
        with self.assertRaises(ValueError):
            t = metis.bisection(A)

        A[1,0] = A[0,1] = NP.float32(NP.inf)
        with self.assertRaises(ValueError):
            t = metis.bisection(A)

        A[1,0] = A[0,1] = NP.float32(NP.nan)
        with self.assertRaises(ValueError):
            t = metis.bisection(A)



    def test_complex(self):
        A = SS.lil_matrix( (4,4), dtype=NP.complex128 )

        with self.assertRaises(ValueError):
            t = metis.bisection(A)



class Test_nested_dissection(unittest.TestCase):
    def check_return_value(self, n, perm, sizes):
        self.assertTrue( isinstance(perm, NP.ndarray) )
        self.assertEqual( perm.shape, (n,) )

        self.assertTrue( isinstance(sizes, NP.ndarray) )
        self.assertEqual( sizes.size, 3 )


    def test_simple(self):
        for n in [1,2,3,4]:
            A = SS.identity(n, dtype=NP.float32, format='csc')
            perm, sizes = metis.nested_dissection(A)

            self.check_return_value(n, perm, sizes)



    def test_arrow(self):
        n = 3
        A = SS.identity(n, dtype=NP.float32, format='lil')
        A[:,0] = 1
        A[0,:] = 1

        perm, sizes = metis.nested_dissection(A)

        self.check_return_value(n, perm, sizes)



if __name__ == '__main__':
    unittest.main()
