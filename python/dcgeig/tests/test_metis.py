#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import unittest

import dcgeig.metis as metis
import numpy as NP
import scipy.sparse as SS



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
