#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2015-2016 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

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
