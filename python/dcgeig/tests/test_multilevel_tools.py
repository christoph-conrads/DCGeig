#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import unittest

import dcgeig.multilevel_tools as tools
import dcgeig.utils as utils
from dcgeig.sparse_tools import Tree

import numpy as NP
import numpy.matlib as ML

import scipy.sparse as SS
import scipy.linalg as SL
import scipy.sparse.linalg as LA



class Test_make_eigenpair_selector(unittest.TestCase):
    def make_options(self, c_s, n_s_min):
        options = tools.get_default_options()
        options.c_s = c_s
        options.n_s_min = n_s_min

        return options



    def test_simple(self):
        options = self.make_options(2, 0)
        lambda_c = 10
        m = 50
        d = NP.arange(m)

        select = tools.make_eigenpair_selector(options, lambda_c, level=1)
        t = select(d)

        self.assertTrue( NP.any(t) )



    def test_min_size(self):
        n_s_min = 8
        options = self.make_options(1, n_s_min)
        lambda_c = 100

        d = NP.arange(101, 150)

        select = tools.make_eigenpair_selector(options, lambda_c, level=1)
        t = select(d)

        self.assertTrue( NP.all(t[:n_s_min]) )


    def test_select_only_finite_values(self):
        n_s_min = 10
        options = self.make_options(1, n_s_min)
        lambda_c = 1.0

        d = NP.array([100, float('inf')])

        select = tools.make_eigenpair_selector(options, lambda_c, level=1)
        t = select(d)

        self.assertTrue( t[0] )
        self.assertFalse( t[1] )


    def test_level_selection(self):
        n = 100
        options = self.make_options(2, 32)
        lambda_c = 1.0
        d = NP.arange(n, dtype=NP.float64)

        f = tools.make_eigenpair_selector(options, lambda_c, level=1)
        t1 = f(d)

        g = tools.make_eigenpair_selector(options, lambda_c, level=100)
        t100 = g(d)

        self.assertTrue( NP.sum(t100) >= NP.sum(t1) )
        self.assertTrue( NP.all( (t1 & t100) == t1 ) )



class Test_rayleigh_ritz(unittest.TestCase):
    def test_simple(self):
        n = 5
        m = 2

        ds = NP.arange(1.0 * n)
        K = SS.spdiags(ds, 0, n, n, format='csc')
        M = SS.identity(n, format='lil')

        S = ML.zeros( [n,m] )
        S[2:4,:] = NP.array( [[1, 1], [1, 0]] )

        d, X, eta, delta = tools.rayleigh_ritz(K, M, S)

        eps = NP.finfo(d.dtype).eps
        self.assertTrue( NP.all(eta <= n * eps) )



if __name__ == '__main__':
    unittest.main()
