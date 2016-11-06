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

import scipy.sparse as SS



class Test_spll(unittest.TestCase):
    def test_simple(self):
        n = 5
        A = SS.diags( [1.0 * NP.arange(1, n+1)], [0], [n,n], format='csc' )

        LL = linalg.spll(A)

        b = 2*3*4*5 * NP.ones([n,1])
        x = LL.solve(b)

        self.assertEqual( NL.norm(A*x - b), 0 )
