#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import unittest

import numpy as NP

import scipy.sparse as SS

import dcgeig.binary_tree as binary_tree
import dcgeig.solver as solver



class Test_estimate_trace(unittest.TestCase):
    def test_simple(self):
        n = 3
        m = n-1
        f = lambda x: x
        b = 2
        P = SS.eye(n, m)
        node = binary_tree.make_leaf_node(m)

        node = solver.estimate_trace(f, b, P, node)

        self.assertEqual( node.trace_mean, m )



class Test_estimate_eigenvalue_count(unittest.TestCase):
    def test_simple(self):
        n = 9
        K = SS.diags(NP.arange(1, n+1), dtype=NP.float64, format='csc')
        M = SS.identity(n)

        left_child = binary_tree.make_leaf_node(4)
        right_child = binary_tree.make_leaf_node(5)
        root = binary_tree.make_internal_node(left_child, right_child, n)
        eps = 0.05

        new_root = solver.estimate_eigenvalue_count(root, K, M, 0.75, 1.5, 50)

        mu = new_root.trace_mean
        std = new_root.trace_std

        self.assertTrue( abs(mu - 1) < eps )
        self.assertTrue( std < mu )



if __name__ == '__main__':
    unittest.main()
