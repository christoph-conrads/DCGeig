#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import unittest

import dcgeig.multilevel_solver as MS
import dcgeig.multilevel_tools as multilevel_tools
from dcgeig.sparse_tools import Tree

import numpy as NP
import scipy.sparse as SS



class Test_impl(unittest.TestCase):
    def test_simple(self):
        n = 2
        K = SS.identity(n, format='csc')
        M = SS.identity(n, format='csc')

        options = multilevel_tools.get_default_options()
        options.lambda_c = 1
        ptree = Tree.make_leaf_node({'n': n})

        d, X, _ = MS.impl(options, K, M, 0, ptree)

        self.assertEqual( d[0], 1 )
        self.assertEqual( d[1], 1 )



    def test_4by4(self):
        n = 4
        K = SS.identity(n, format='csc')
        M = SS.identity(n, format='csc')

        options = multilevel_tools.get_default_options()
        options.lambda_c = 1
        options.n_direct = 2
        ptree = \
            Tree.make_internal_node( \
                Tree.make_leaf_node({'n': 2}),
                Tree.make_leaf_node({'n': 2}),
                {'n': n}
            )

        d, X, _ = MS.impl(options, K, M, 0, ptree)

        self.assertTrue( NP.all(d == 1) )



    def test_returns_only_finite_eigenvalues(self):
        n = 2
        K = SS.identity(n, format='csc')
        M = SS.spdiags([0,1.0], 0, n, n, format='csc')

        options = multilevel_tools.get_default_options()
        options.lambda_c = 1
        ptree = Tree.make_leaf_node({'n': n})

        d, X, _ = MS.impl(options, K, M, 0, ptree)

        self.assertEqual( d.size, 1 )
        self.assertEqual( d[0], 1 )



class Test_execute(unittest.TestCase):
    def test_simple(self):
        n = 6
        dtype = NP.float32

        M = SS.identity(n, dtype=dtype, format='csc')
        K = SS.csc_matrix( \
                NP.array([ \
                    [50, 1, 1, 1, 1, 1],
                    [ 1,20,-1, 0, 0, 0],
                    [ 1,-1,20,-1, 0, 0],
                    [ 1, 0,-1,20, 0, 0],
                    [ 1, 0, 0, 0,10, 1],
                    [ 1, 0, 0, 0, 1,10]], dtype=dtype))

        options = multilevel_tools.get_default_options()
        options.lambda_c = 1e-8
        options.K = K
        options.M = M

        d, eta, _ = MS.execute(options)

        self.assertTrue( NP.all(eta < 2*NP.finfo(dtype).eps) )



if __name__ == '__main__':
    unittest.main()
