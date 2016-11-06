#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

import dcgeig.utils as utils

import scipy.sparse as SS
import scipy.sparse.linalg as LA



class SuperLL:
    def __init__(self, LU):
        self.LU = LU
        self.shape = LU.shape
        self.matvec = LU.solve
        self.solve = LU.solve


def spll(A):
    assert SS.isspmatrix(A)
    assert utils.is_hermitian(A)

    options = {'SymmetricMode': True}
    LU = LA.splu(A, diag_pivot_thresh=0.0, options=options)

    return SuperLL(LU)
