#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

import numbers

import numpy as NP
import numpy.matlib as ML

import scipy.linalg as SL
import scipy.sparse as SS



def fdm_laplacian_1D(n, l=1.0, dtype=NP.float64, format='csc'):
    assert isinstance(n, int)
    assert n > 0
    assert isinstance(l, numbers.Real)
    assert l > 0

    h = 1.0 * l / (n+1)

    if n < 3:
        T = SS.kron(1, [[2, -1],[-1, 2]], format=format)
        A = T[:n,:][:,:n]
        return (A/h**2).astype(dtype)

    xs = NP.ones(n)
    A = SS.diags([-xs, 2*xs, -xs], [-1,0,+1], [n,n], format=format)

    return (A/h**2).astype(dtype)


def fdm_laplacian_2D(n, dtype=NP.float64, format='csc'):
    return fdm_laplacian([n,n], [1.0,1.0], dtype=dtype, format=format)


def fdm_laplacian(ns, ls, dtype=NP.float64, format='csc'):
    assert len(ns) == len(ls)
    assert len(ns) > 0
    assert isinstance(ns[-1], int)
    assert ns[-1] > 0
    assert isinstance(ls[-1], numbers.Real)
    assert ls[-1] > 0
    assert isinstance(format, str)

    n = ns[-1]
    l = ls[-1]
    d = len(ns)

    if d == 1:
        return fdm_laplacian_1D(n, l, dtype, format)

    A0 = fdm_laplacian_1D(n, l, dtype, format)
    Ad = fdm_laplacian(ns[:-1], ls[:-1], dtype, format)

    K = SS.kronsum(Ad, A0, format=format)

    assert K.dtype == dtype

    return K



def fdm_laplacian_1D_eigenpairs(n, l=1.0, dtype=NP.float64):
    assert isinstance(n, int)
    assert n > 0
    assert isinstance(l, numbers.Real)
    assert l > 0

    h = 1.0 * l / (n+1)

    i = NP.reshape(NP.arange(1,n+1), [n,1])
    j = NP.reshape(NP.arange(1,n+1), [1,n])

    X = NP.sin( NP.pi * i * j / (n+1) )
    d = 1 - NP.cos( NP.pi * NP.arange(1,n+1) / (n+1) )

    return (2/h**2 * d).astype(dtype), ML.matrix(X, dtype=dtype)


def fdm_laplacian_eigenpairs_impl(ns, ls, dtype):
    assert len(ns) == len(ls)
    assert len(ns) > 0
    assert isinstance(ns[-1], int)
    assert ns[-1] > 0
    assert isinstance(ls[-1], numbers.Real)
    assert ls[-1] > 0

    n = ns[-1]
    l = ls[-1]
    dim = len(ns)

    if dim == 1:
        return fdm_laplacian_1D_eigenpairs(n, l, dtype)

    d, U = fdm_laplacian_eigenpairs_impl(ns[:-1], ls[:-1], dtype)
    e, V = fdm_laplacian_1D_eigenpairs(n, l, dtype)

    m = d.size
    f = NP.reshape(e.reshape([n,1]) + d.reshape([1,m]), m*n)
    W = SL.kron(V, U)

    return f, W


def fdm_laplacian_eigenpairs(ns, ls, dtype=NP.float64):
    d, U = fdm_laplacian_eigenpairs_impl(ns, ls, dtype)
    U = U / SL.norm(U, axis=0)

    assert d.dtype == dtype
    assert U.dtype == dtype

    i = NP.argsort(d)

    return d[i], ML.matrix(U[:,i])



def fem_laplacian_1D(n, l=1.0, dtype=NP.float64, format='csc'):
    assert isinstance(n, int)
    assert n > 0
    assert isinstance(l, numbers.Real)
    assert l > 0

    h = l / (n+1)

    if n < 3:
        A = SS.kron(1, [[2, -1],[-1, 2]], format=format)
        K = A[:n,:][:,:n]

        B = SS.kron(1, [[4, 1], [1, 4]], format=format)
        M = B[:n,:][:,:n]

        return (K/h).astype(dtype), (h/6*M).astype(dtype)


    xs = NP.ones(n)
    A = SS.diags( [-xs, 2*xs, -xs], [-1,0,+1], [n,n], format=format )
    B = SS.diags( [+xs, 4*xs, +xs], [-1,0,+1], [n,n], format=format )

    return (A/h).astype(dtype), (h/6*B).astype(dtype)


def fem_laplacian_2D(n, dtype=NP.float64, format='csc'):
    return fem_laplacian([n,n], [1.0,1.0], dtype=dtype, format=format)


def fem_laplacian(ns, ls, dtype=NP.float64, format='csc'):
    assert len(ns) == len(ls)
    assert isinstance(ns[-1], int)
    assert ns[-1] > 0
    assert isinstance(ls[-1], numbers.Real)
    assert ls[-1] > 0

    n = ns[-1]
    l = ls[-1]
    dim = len(ns)

    K0, M0 = fem_laplacian_1D(n, l, dtype, format)

    if dim == 1:
        return K0, M0

    Kd, Md = fem_laplacian(ns[:-1], ls[:-1], dtype, format)

    K = SS.kron(M0, Kd, format=format) + SS.kron(K0, Md, format=format)
    M = SS.kron(M0, Md, format=format)

    assert K.dtype == dtype
    assert K.getformat() == format
    assert M.dtype == dtype
    assert M.getformat() == format

    return K, M



def fem_laplacian_1D_eigenpairs(n, l=1.0, dtype=NP.float64):
    assert isinstance(n, int)
    assert n > 0
    assert isinstance(l, numbers.Real)
    assert l > 0

    h = 1.0 * l / (n+1)

    i = NP.reshape(NP.arange(1,n+1), [n,1])
    j = NP.reshape(NP.arange(1,n+1), [1,n])

    X = NP.sin( NP.pi * i * j / (n+1) )

    nominator = 1 - NP.cos( NP.pi * NP.arange(1,n+1) / (n+1) )
    denominator = 2 + NP.cos( NP.pi * NP.arange(1,n+1) / (n+1) )
    d = 6/h**2 * nominator / denominator

    return d.astype(dtype), ML.matrix(X, dtype=dtype)


def fem_laplacian_eigenpairs_impl(ns, ls, dtype):
    assert len(ns) == len(ls)
    assert len(ns) > 0
    assert isinstance(ns[-1], int)
    assert ns[-1] > 0
    assert isinstance(ls[-1], numbers.Real)
    assert ls[-1] > 0

    n = ns[-1]
    l = ls[-1]
    dim = len(ns)

    if dim == 1:
        return fem_laplacian_1D_eigenpairs(n, l, dtype)

    d, U = fem_laplacian_eigenpairs_impl(ns[:-1], ls[:-1], dtype)
    e, V = fem_laplacian_1D_eigenpairs(n, l, dtype)

    m = d.size
    f = NP.reshape(e.reshape([n,1]) + d.reshape([1,m]), m*n)
    W = SL.kron(V, U)

    return f, W


def fem_laplacian_eigenpairs(ns, ls, dtype=NP.float64):
    d, U = fem_laplacian_eigenpairs_impl(ns, ls, dtype)
    U = U / SL.norm(U, axis=0)

    assert d.dtype == dtype
    assert U.dtype == dtype

    i = NP.argsort(d)

    return d[i], ML.matrix(U[:,i])
