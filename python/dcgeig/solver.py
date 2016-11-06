#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import numbers

import copy

import numpy as NP
import numpy.random

import scipy.sparse as SS

import dcgeig.polynomial as polynomial



def estimate_trace(f, n, b, dtype=NP.float64):
    assert callable(f)
    assert isinstance(n, int)
    assert n > 0
    assert isinstance(b, int)
    assert b > 0
    assert b <= n


    random = numpy.random.RandomState(seed=1)

    V = 2 * random.randint(0, 2, [n,b]).astype(dtype) - 1
    W = f(V)
    xs = NP.einsum('ij,ij->j', V, W, casting='no')

    mean = NP.mean(xs)
    std = NP.std(xs, ddof=1)

    return mean, std



def estimate_eigenvalue_count(K, M, lambda_1, lambda_c, d, b):
    assert SS.isspmatrix(K)
    assert SS.isspmatrix(M)
    assert isinstance(lambda_1, numbers.Real)
    assert lambda_1 > 0
    assert isinstance(lambda_c, numbers.Real)
    assert lambda_c > lambda_1
    assert isinstance(d, int)
    assert d > 0
    assert isinstance(b, int)
    assert b > 0

    n = K.shape[0]
    f = polynomial.approximate_projection(d, lambda_1, lambda_c, K, M)

    mean, std = estimate_trace(f, n, b, K.dtype)

    return mean, std
