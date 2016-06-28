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



class Test_bound_expected_backward_error(unittest.TestCase):
    def test_simple(self):
        options = tools.Options()
        options.lambda_c = 1
        n = 5
        p = 3
        m = n-p

        K = SS.identity(n, format='lil')
        K21 = SS.csc_matrix( (m,p) )

        M = SS.identity(n, format='dok')
        M21 = SS.lil_matrix( (m,p) )

        eta = tools.bound_expected_backward_error(options, K, K21, M, M21)

        self.assertTrue( NP.isrealobj(eta) )
        self.assertEqual( eta, 0 )



    def test_cases(self):
        options = tools.Options()
        options.lambda_c = 10
        n = 7
        p = 4
        m = n-p
        assert p >= m

        ds = NP.full(n, 1, dtype=NP.complex128)
        A = SS.spdiags(ds, 0, n, n, format='lil')
        A[n-1,0] = 0.0 + 1.0j
        A[0,n-1] = 0.0 - 1.0j
        A21 = A[:,:p][-m:,:]

        B = SS.identity(n, dtype=NP.complex128, format='dok')
        B21 = B[:,:p][-m:,:]

        eta = tools.bound_expected_backward_error(options, A, A21, B, B21)

        self.assertTrue( NP.isrealobj(eta) )
        self.assertEqual( eta, NP.sqrt(1.5/p) * 1.0/3.0 )

        eta = tools.bound_expected_backward_error(options, B, B21, A, A21)

        self.assertTrue( NP.isrealobj(eta) )
        self.assertTrue( eta > 0 )



class Test_make_eigenpair_selector(unittest.TestCase):
    class Options:
        def __init__(self, lambda_c, c_s, n_s_min):
            self.lambda_c = lambda_c
            self.c_s = c_s
            self.n_s_min = n_s_min



    def test_simple(self):
        options = self.Options(10, 2, 0)
        m = 50
        d = NP.arange(m)
        delta = NP.zeros(m)

        select = tools.make_eigenpair_selector(options, level=1)
        t = select(d, delta)

        self.assertTrue( NP.any(t) )



    def test_min_size(self):
        n_s_min = 8
        options = self.Options(100, 1, n_s_min)

        d = NP.arange(101, 150)
        delta = NP.zeros(d.size)

        select = tools.make_eigenpair_selector(options, level=1)
        t = select(d, delta)

        self.assertTrue( NP.all(t[:n_s_min]) )


    def test_select_only_finite_values(self):
        n_s_min = 10
        options = self.Options(1, 1, n_s_min)

        d = NP.array([100, float('inf')])
        delta = NP.zeros(d.size)

        select = tools.make_eigenpair_selector(options, level=1)
        t = select(d, delta)

        self.assertTrue( t[0] )
        self.assertFalse( t[1] )


    def test_level_selection(self):
        n = 100
        options = self.Options(1, 2, 32)
        d = NP.arange(n, dtype=NP.float64)
        delta = NP.ones(d.size)

        f = tools.make_eigenpair_selector(options, level=1)
        t1 = f(d, delta)

        g = tools.make_eigenpair_selector(options, level=100)
        t100 = g(d, delta)

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
