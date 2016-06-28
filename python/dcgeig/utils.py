#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2015-2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import numpy as NP
import numpy.matlib as ML
import scipy.sparse as SS



def force_hermiticity(A):
    assert type(A) == ML.matrix

    return ML.matrix(NP.triu(A)) + ML.matrix(NP.triu(A, 1)).H



def is_hermitian(A):
    assert len(A.shape) == 2

    if A.shape[0] != A.shape[1]:
        return False

    if SS.isspmatrix(A):
        return (A != A.H).nnz == 0
    elif type(A) is ML.matrix:
        return NP.all(A == A.H)
    else:
        raise ValueError('Unknown matrix type')
