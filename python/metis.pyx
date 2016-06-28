# Copyright 2016 Christoph Conrads
#
# This file is part of DCGeig and it is subject to the terms of the DCGeig
# license. See http://DCGeig.tech/license for a copy of this license.

cdef extern from "metis.h":
    ctypedef long int idx_t
    ctypedef float real_t

    cdef int IDXTYPEWIDTH
    cdef int REALTYPEWIDTH

    cdef int METIS_NOPTIONS
    cdef int METIS_OPTION_NUMBERING

    cdef int METIS_OK
    cdef int METIS_ERROR_INPUT
    cdef int METIS_ERROR_MEMORY

    int METIS_SetDefaultOptions(idx_t* p_options)

    int METIS_NodeNDP(
        idx_t n_vertices, idx_t* p_i, idx_t* p_j, idx_t* p_vertex_weights,
        idx_t npes, idx_t* options, idx_t* perm, idx_t* iperm, idx_t* sizes)


import ctypes

import numpy as NP
import scipy.sparse as SS
import dcgeig.utils as utils

cimport cython
cimport numpy as NP



@cython.overflowcheck(True)
def sanitize_input(A not None):
    assert IDXTYPEWIDTH == 64

    if not SS.isspmatrix(A):
        raise ValueError('A must be a sparse matrix')

    cdef idx_t n = A.shape[0]

    if n == 0:
        raise ValueError('Matrix must not be empty')
    if not utils.is_hermitian(A):
        raise ValueError('Matrix must be symmetric')
    if NP.isinf(A.sum()) or NP.isnan(A.sum()):
        raise ValueError('Matrix entries must be finite real values')


    # remove diagonal entries / self-loops
    A = SS.lil_matrix(A, copy=True)
    A.setdiag(0)
    A = SS.csc_matrix(A)
    # is this done when constructing the CSC matrix?
    A.sum_duplicates()
    A.eliminate_zeros()
    A.sort_indices()

    return A



@cython.overflowcheck(True)
def check_return_value(ret):
    assert ret != METIS_ERROR_INPUT

    if ret == METIS_OK:
        return

    if ret == METIS_ERROR_MEMORY:
        raise MemoryError('METIS ran out of memory')

    raise RuntimeError('METIS signaled an error (ret={0})'.format(ret))



@cython.overflowcheck(True)
def nested_dissection(A not None):
    A = sanitize_input(A)
    cdef idx_t n = A.shape[0]

    cdef idx_t npes = 2
    cdef NP.ndarray[idx_t, ndim=1] perm = NP.full(n, -1, ctypes.c_long)
    cdef NP.ndarray[idx_t, ndim=1] sizes = NP.full(2*npes-1, -1, ctypes.c_long)

    if A.nnz == 0:
        perm = NP.arange(n)
        sizes = NP.array([n/2, n-n/2, 0])
        return perm, sizes


    cdef NP.ndarray[idx_t, ndim=1] p_i = A.indptr.astype(ctypes.c_long)
    cdef NP.ndarray[idx_t, ndim=1] p_j = A.indices.astype(ctypes.c_long)
    assert p_i.size == n+1
    assert p_j.size == A.nnz

    cdef NP.ndarray[idx_t, ndim=1] options = \
        NP.full(METIS_NOPTIONS, 0, dtype=ctypes.c_long)
    METIS_SetDefaultOptions(&options[0])
    options[METIS_OPTION_NUMBERING] = 0

    cdef NP.ndarray[idx_t, ndim=1] iperm = NP.full(n, -1, ctypes.c_long)

    cdef int ret = METIS_NodeNDP(
        n, &p_i[0], &p_j[0], NULL,
        npes, &options[0], &perm[0], &iperm[0], &sizes[0])

    assert NP.all( NP.sort(perm) == NP.arange(n) )
    assert NP.all( NP.sort(iperm) == NP.arange(n) )
    assert NP.all( perm[iperm] == NP.arange(n) )
    assert NP.sum(sizes) == n

    return perm, sizes
