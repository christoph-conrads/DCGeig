#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import numbers

import numpy as NP

import scipy.sparse as SS



def fdm_laplacian_1D(n, l=1.0, format='csc'):
    assert isinstance(n, int)
    assert n > 0
    assert isinstance(l, numbers.Real)
    assert l > 0

    h = l / (n+1)

    xs = NP.ones(n)
    A = SS.diags( [-xs, 2*xs, -xs], [-1,0,+1], [n,n], format=format )

    return A/h**2



def fdm_laplacian_2D(n, format='csc'):
    A = fdm_laplacian_1D(n, format=format)
    I = SS.identity(n)

    K = SS.kron(I, A, format=format) + SS.kron(A, I, format=format)

    return K



def fem_laplacian_1D(n, l=1.0, format='csc'):
    assert isinstance(n, int)
    assert n > 0
    assert isinstance(l, numbers.Real)
    assert l > 0

    h = l / (n+1)

    xs = NP.ones(n)
    A = SS.diags( [-xs, 2*xs, -xs], [-1,0,+1], [n,n], format=format )
    B = SS.diags( [+xs, 4*xs, +xs], [-1,0,+1], [n,n], format=format )

    return A/h, h/6 * B



def fem_laplacian_2D(n, format='csc'):
    assert isinstance(n, int)
    assert n > 0

    A, B = fem_laplacian_1D(n, format=format)

    K = SS.kron(A, B, format=format) + SS.kron(B, A, format=format)
    M = SS.kron(B, B, format=format)

    return K, M



def fem_laplacian_2D_rectangle(n1, l1, n2, l2, format='csc'):
    A1, B1 = fem_laplacian_1D(n1, l1, format=format)
    A2, B2 = fem_laplacian_1D(n2, l2, format=format)

    K = SS.kron(A1, B2, format=format) + SS.kron(B1, A2, format=format)
    M = SS.kron(B1, B2, format=format)

    return K, M
