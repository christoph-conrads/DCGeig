# Copyright 2015-2016 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

cdef extern from "hpsd_gep_solvers.h":
    ctypedef int lapack_int

    lapack_int solve_gep_with_qr_csd_single(
            lapack_int n,
            float* K, lapack_int ldk,
            float* M, lapack_int ldm,
            float* theta, lapack_int* rank,
            float* work, lapack_int lwork,
            lapack_int* iwork, lapack_int liwork);

    lapack_int solve_gep_with_qr_csd_double(
            lapack_int n,
            double* K, lapack_int ldk,
            double* M, lapack_int ldm,
            double* theta, lapack_int* rank,
            double* work, lapack_int lwork,
            lapack_int* iwork, lapack_int liwork);



    lapack_int solve_gep_with_gsvd_single(
            lapack_int n,
            float* K, lapack_int ldk,
            float* M, lapack_int ldm,
            float* lam, lapack_int* rank,
            float* workspace, lapack_int workspace_size,
            lapack_int* iwork, lapack_int liwork);

    lapack_int solve_gep_with_gsvd_double(
            lapack_int n,
            double* K, lapack_int ldk,
            double* M, lapack_int ldm,
            double* lam, lapack_int* rank,
            double* workspace, lapack_int workspace_size,
            lapack_int* iwork, lapack_int liwork);



    lapack_int deflate_gep_single(
            lapack_int n,
            float* K, lapack_int ldk,
            float* M, lapack_int ldm,
            float* theta, lapack_int* rank_M,
            float* X, lapack_int ldx,
            float* Q, lapack_int ldq,
            float* work, lapack_int lwork,
            lapack_int* iwork, lapack_int liwork);

    lapack_int deflate_gep_double(
            lapack_int n,
            double* K, lapack_int ldk,
            double* M, lapack_int ldm,
            double* theta, lapack_int* rank_M,
            double* X, lapack_int ldx,
            double* Q, lapack_int ldq,
            double* work, lapack_int lwork,
            lapack_int* iwork, lapack_int liwork);



    lapack_int solve_gep_with_deflation_single(
            lapack_int n,
            float* K, lapack_int ldk,
            float* M, lapack_int ldm,
            float* theta, lapack_int* p_rank_M,
            float* work, lapack_int lwork,
            lapack_int* iwork, lapack_int liwork);

    lapack_int solve_gep_with_deflation_double(
            lapack_int n,
            double* K, lapack_int ldk,
            double* M, lapack_int ldm,
            double* theta, lapack_int* p_rank_M,
            double* work, lapack_int lwork,
            lapack_int* iwork, lapack_int liwork);



import ctypes

import numpy as N
import numpy.matlib as matlib
import numpy.linalg as NL

import scipy.linalg as LA

import utils as U

cimport numpy as N
cimport cython


@cython.overflowcheck(True)
def check_input( \
    N.ndarray K not None,
    N.ndarray M not None):

    if not type(K) == matlib.matrix or not type(M) == matlib.matrix:
        raise TypeError("Both arguments must be matrices")

    if K.shape[0] != K.shape[1] or M.shape[0] != M.shape[1]:
        raise ValueError("Matrices must be square")

    if K.shape[0] != M.shape[0]:
        raise ValueError("Matrices must have the same dimension")

    if not U.is_hermitian(K) or not U.is_hermitian(M):
        raise ValueError("Matrices must be hermitian")

    if K.size == 0:
        raise ValueError("Matrices must not be empty")

    if K.dtype != M.dtype:
        raise ValueError("Both matrices must have the same dtype")

    if K.dtype not in [ctypes.c_float, ctypes.c_double]:
        raise ValueError("dtype must be float32 or float64")



# standard solver
@cython.overflowcheck(True)
def standard( \
    N.ndarray K not None,
    N.ndarray M not None):
    check_input(K, M)

    theta, X = LA.eigh(K, M)
    X = matlib.matrix(X)

    return theta, X



# standard solver for float problems with mass matrix shift
def sigma32(
    N.ndarray[float, ndim=2] K not None,
    N.ndarray[float, ndim=2] M not None):
    check_input(K, M)

    if NL.norm(K) == 0 or NL.norm(M) == 0:
        return standard(K, M)

    cdef int n = len(K)

    cdef N.ndarray[double, ndim=2] A = K.astype(N.float64)
    cdef N.ndarray[double, ndim=2] B = M.astype(N.float64)

    cdef double normA_F = NL.norm(A, 'fro')
    cdef double normB_F = NL.norm(B, 'fro')
    cdef double s = normA_F / normB_F

    B = s*B + normA_F * N.finfo(N.float32).eps * matlib.eye(n)

    cdef N.ndarray theta
    cdef N.ndarray X
    theta, X = LA.eigh(A, B)

    theta = (s * theta).astype(K.dtype)
    X = matlib.matrix( X.astype(K.dtype) )

    return theta, X



# QR+CSD GEP solver
@cython.overflowcheck(True)
def qr_csd( \
    N.ndarray K not None,
    N.ndarray M not None):
    check_input(K, M)

    K = matlib.matrix(K)
    M = matlib.matrix(M)

    cdef lapack_int ret = -1;
    if K.dtype == ctypes.c_float:
        ret, X, theta = call_solve_gep_with_qr_csd_single(K, M)
    elif K.dtype == ctypes.c_double:
        ret, X, theta = call_solve_gep_with_qr_csd_double(K, M)
    else:
        raise ValueError("dtype must be float32 or float64")

    if ret == 1:
        raise LA.LinAlgError('K is not positive semidefinite')
    if ret == 2:
        raise LA.LinAlgError('M is not positive semidefinite')
    if ret == 3:
        raise LA.LinAlgError('xORCSD2BY1 failed to converge')
    assert ret == 0

    return theta, X



def call_solve_gep_with_qr_csd_single( \
    N.ndarray[float, ndim=2] K not None,
    N.ndarray[float, ndim=2] M not None):

    dtype = ctypes.c_float
    cdef lapack_int n = K.shape[0]
    cdef lapack_int ldk = n
    cdef lapack_int ldm = n

    cdef N.ndarray[float, ndim=1] theta = matlib.full(n, N.nan, dtype)
    cdef int rank = -1

    cdef lapack_int lwork = qr_csd_workspace_query(n)
    cdef N.ndarray[float, ndim=1] work = matlib.full(lwork, N.nan, dtype)

    cdef lapack_int liwork = 3*n
    cdef N.ndarray[lapack_int, ndim=1] iwork = \
        matlib.full(liwork, 0, dtype=ctypes.c_int)

    cdef lapack_int ret = solve_gep_with_qr_csd_single(
        n, &K[0,0], ldk, &M[0,0], ldm, &theta[0], &rank,
        &work[0], lwork, &iwork[0], liwork)

    return ret, K, theta



def call_solve_gep_with_qr_csd_double( \
    N.ndarray[double, ndim=2] K not None,
    N.ndarray[double, ndim=2] M not None):

    dtype = ctypes.c_double
    cdef lapack_int n = K.shape[0]
    cdef lapack_int ldk = n
    cdef lapack_int ldm = n

    cdef N.ndarray[double, ndim=1] theta = matlib.full(n, N.nan, dtype)
    cdef int rank = -1

    cdef lapack_int lwork = qr_csd_workspace_query(n)
    cdef N.ndarray[double, ndim=1] work = matlib.full(lwork, N.nan, dtype)

    cdef lapack_int liwork = 3*n
    cdef N.ndarray[lapack_int, ndim=1] iwork = \
        matlib.full( liwork, 0, dtype=ctypes.c_int )

    cdef lapack_int ret = solve_gep_with_qr_csd_double(
        n, &K[0,0], ldk, &M[0,0], ldm, &theta[0], &rank,
        &work[0], lwork, &iwork[0], liwork)

    return ret, K, theta



def qr_csd_workspace_query(lapack_int n):
    assert n > 0

    cdef double opt = N.nan

    cdef lapack_int ret = solve_gep_with_qr_csd_double( \
        n, NULL, n, NULL, n, NULL, NULL, &opt, -1, NULL, n)
    assert ret == 0

    assert opt > 0
    assert round(opt) == opt

    return int(opt)



# GSVD GEP solver
@cython.overflowcheck(True)
def gsvd( \
    N.ndarray K not None,
    N.ndarray M not None):
    check_input(K, M)

    # copy matrices because the GSVD solver overwrites them
    K = K.astype( K.dtype, order='F' )
    M = matlib.matrix(M)


    cdef lapack_int ret = -1;
    if K.dtype == ctypes.c_float:
        ret, theta = call_solve_gep_with_gsvd_single(K, M)
    elif K.dtype == ctypes.c_double:
        ret, theta = call_solve_gep_with_gsvd_double(K, M)
    else:
        raise ValueError("dtype must be float32 or float64")

    if ret == 1:
        raise LA.LinAlgError('K is not positive semidefinite')
    if ret == 2:
        raise LA.LinAlgError('M is not positive semidefinite')
    if ret == 3:
        raise LA.LinAlgError('xGGSVD3 failed to converge')
    assert ret == 0

    return theta, K



def call_solve_gep_with_gsvd_single( \
    N.ndarray[float, ndim=2] K not None,
    N.ndarray[float, ndim=2] M not None):

    dtype = ctypes.c_float
    cdef lapack_int n = K.shape[0]
    cdef lapack_int ldk = n
    cdef lapack_int ldm = n

    cdef N.ndarray[float, ndim=1] theta = matlib.full( n, N.nan, dtype )
    cdef lapack_int rank = -1

    cdef lapack_int workspace_size = gsvd_workspace_query(n)
    cdef N.ndarray[float, ndim=1] workspace = \
        matlib.full( workspace_size, N.nan, dtype )

    cdef lapack_int liwork = n
    cdef N.ndarray[lapack_int, ndim=1] iwork = \
        matlib.full( liwork, 0, dtype=ctypes.c_int )

    cdef lapack_int ret = solve_gep_with_gsvd_single(
        n, &K[0,0], ldk, &M[0,0], ldm, &theta[0], &rank,
        &workspace[0], workspace_size, &iwork[0], liwork)

    return ret, theta



def call_solve_gep_with_gsvd_double( \
    N.ndarray[double, ndim=2] K not None,
    N.ndarray[double, ndim=2] M not None):

    dtype = ctypes.c_double
    cdef lapack_int n = K.shape[0]
    cdef lapack_int ldk = n
    cdef lapack_int ldm = n

    cdef N.ndarray[double, ndim=1] theta = matlib.full( n, N.nan, dtype )
    cdef lapack_int rank = -1

    cdef lapack_int workspace_size = gsvd_workspace_query(n)
    cdef N.ndarray[double, ndim=1] workspace = \
        matlib.full( workspace_size, N.nan, dtype )

    cdef lapack_int liwork = n
    cdef N.ndarray[lapack_int, ndim=1] iwork = \
        matlib.full( liwork, 0, dtype=ctypes.c_int )

    cdef lapack_int ret = solve_gep_with_gsvd_double(
        n, &K[0,0], ldk, &M[0,0], ldm, &theta[0], &rank,
        &workspace[0], workspace_size, &iwork[0], liwork)

    return ret, theta



def gsvd_workspace_query(lapack_int n):
    assert n > 0

    cdef double opt = N.nan

    cdef lapack_int ret = solve_gep_with_gsvd_double( \
        n, NULL, n, NULL, n, NULL, NULL, &opt, -1, NULL, n)
    assert ret == 0

    assert opt > 0
    assert round(opt) == opt

    return int(opt)



# GEP deflation
def deflate_gep( \
    N.ndarray K not None,
    N.ndarray M not None):
    check_input(K, M)

    K = K.astype( K.dtype, order='F' )
    M = M.astype( M.dtype, order='F' )

    cdef lapack_int ret = -1;
    if K.dtype == ctypes.c_float:
        ret, rank_M, X, Q = call_deflate_gep_single(K, M)
    elif K.dtype == ctypes.c_double:
        ret, rank_M, X, Q = call_deflate_gep_double(K, M)
    else:
        raise ValueError("dtype must be float32 or float64")


    assert ret >= 0

    if ret == 1:
        raise ValueError('LAPACK function xSYEVD did not converge')
    if ret == 2:
        raise ValueError('The mass matrix is indefinite')
    if ret == 3:
        raise ValueError('LAPACK function xGESVD did not converge')
    if ret == 4:
        raise ValueError('The matrix pencil is singular')
    if ret != 0:
        raise RuntimeError('deflate_gep returned {0}'.format(ret))


    assert rank_M >= 0
    assert rank_M <= K.shape[0]

    cdef lapack_int r = rank_M
    cdef N.ndarray A = U.force_hermiticity( K[0:r,0:r] )
    cdef N.ndarray B = U.force_hermiticity( M[0:r,0:r] )

    return A, B, X, Q



@cython.overflowcheck(True)
def call_deflate_gep_single( \
    N.ndarray[float, ndim=2] K not None,
    N.ndarray[float, ndim=2] M not None):

    cdef lapack_int n = K.shape[0]
    cdef lapack_int rank_M = -1

    cdef N.ndarray[float, ndim=1] theta = matlib.full(n, N.nan, ctypes.c_float)

    cdef N.ndarray[float, ndim=2] X = matlib.full_like( K, N.nan, order='F' )
    cdef N.ndarray[float, ndim=2] Q = matlib.full_like( K, N.nan, order='F' )

    cdef lapack_int lwork
    cdef lapack_int liwork
    lwork, liwork = deflate_gep_workspace_query(n)

    cdef N.ndarray[float, ndim=1] work = N.full( lwork, N.nan, ctypes.c_float )

    cdef N.ndarray[lapack_int, ndim=1] iwork = \
        N.full( liwork, -1, dtype=ctypes.c_int )


    cdef lapack_int ret = deflate_gep_single( \
        n, &K[0,0], n, &M[0,0], n, &theta[0], &rank_M, &X[0,0], n, &Q[0,0], n,
        &work[0], lwork, &iwork[0], liwork)

    if ret != 0:
        return ret, None, None, None

    return ret, rank_M, X, Q



@cython.overflowcheck(True)
def call_deflate_gep_double( \
    N.ndarray[double, ndim=2] K not None,
    N.ndarray[double, ndim=2] M not None):

    cdef lapack_int n = K.shape[0]
    cdef lapack_int rank_M = -1

    cdef N.ndarray[double, ndim=1] theta = matlib.full(n,N.nan,ctypes.c_double)

    cdef N.ndarray[double, ndim=2] X = matlib.full_like( K, N.nan, order='F' )
    cdef N.ndarray[double, ndim=2] Q = matlib.full_like( K, N.nan, order='F' )

    cdef lapack_int lwork
    cdef lapack_int liwork
    lwork, liwork = deflate_gep_workspace_query(n)

    cdef N.ndarray[double, ndim=1] work = N.full(lwork, N.nan, ctypes.c_double)

    cdef N.ndarray[lapack_int, ndim=1] iwork = \
        N.full( liwork, -1, dtype=ctypes.c_int )

    cdef lapack_int ret = deflate_gep_double( \
        n, &K[0,0], n, &M[0,0], n, &theta[0], &rank_M, &X[0,0], n, &Q[0,0], n,
        &work[0], lwork, &iwork[0], liwork)

    if ret != 0:
        return ret, None, None, None

    return ret, rank_M, X, Q



def deflate_gep_workspace_query(lapack_int n):
    assert n > 0

    cdef double opt = N.nan
    cdef lapack_int iopt = -1

    cdef lapack_int ret = deflate_gep_double( \
        n, NULL, n, NULL, n, NULL, NULL, NULL, n, NULL, n, &opt, -1, &iopt, n)
    assert ret == 0

    assert opt > 0
    assert round(opt) == opt

    return int(opt), iopt



# solve GEP with deflation
@cython.overflowcheck(True)
def deflation( \
    N.ndarray K not None,
    N.ndarray M not None):
    check_input(K, M)

    K = K.astype( K.dtype, order='F' )
    M = M.astype( M.dtype, order='F' )

    cdef lapack_int ret = -1;
    if K.dtype == ctypes.c_float:
        ret, X, lam = call_solve_gep_with_deflation_single(K, M)
    elif K.dtype == ctypes.c_double:
        ret, X, lam = call_solve_gep_with_deflation_double(K, M)
    else:
        raise ValueError("dtype must be float32 or float64")
    assert ret >= 0

    if ret == 1:
        raise LA.LinAlgError('xSYEVD failed to converge')
    if ret == 2:
        raise LA.LinAlgError('Mass matrix not positive definite')
    if ret == 3:
        raise LA.LinAlgError('xGESVD failed to converge')
    if ret == 4:
        raise LA.LinAlgError('Matrix pencil not regular')
    if ret == 5:
        raise LA.LinAlgError('Deflated leading minor not positive definite')
    if ret == 6:
        raise LA.LinAlgError('xSYGVD failed to converge')
    assert ret == 0

    return lam, X



def call_solve_gep_with_deflation_single( \
    N.ndarray[float, ndim=2] K not None,
    N.ndarray[float, ndim=2] M not None):

    dtype = ctypes.c_float
    cdef lapack_int n = K.shape[0]
    cdef lapack_int ldk = n
    cdef lapack_int ldm = n

    cdef N.ndarray[float, ndim=1] theta = matlib.full(n, N.nan, dtype)
    cdef lapack_int rank_M = -1

    cdef lapack_int lwork, liwork
    lwork, liwork = deflation_workspace_query(n)

    cdef N.ndarray[float, ndim=1] work = matlib.full( lwork, N.nan, dtype )

    cdef N.ndarray[lapack_int, ndim=1] iwork = \
        matlib.full( liwork, 0, dtype=ctypes.c_int )

    cdef lapack_int ret = solve_gep_with_deflation_single(
        n, &K[0,0], ldk, &M[0,0], ldm, &theta[0], &rank_M,
        &work[0], lwork, &iwork[0], liwork)

    return ret, K, theta



def call_solve_gep_with_deflation_double( \
    N.ndarray[double, ndim=2] K not None,
    N.ndarray[double, ndim=2] M not None):

    dtype = ctypes.c_double
    cdef lapack_int n = K.shape[0]
    cdef lapack_int ldk = n
    cdef lapack_int ldm = n

    cdef N.ndarray[double, ndim=1] theta = matlib.full(n, N.nan, dtype)
    cdef lapack_int rank_M = -1

    cdef lapack_int lwork, liwork
    lwork, liwork = deflation_workspace_query(n)

    cdef N.ndarray[double, ndim=1] work = matlib.full( lwork, N.nan, dtype )

    cdef N.ndarray[lapack_int, ndim=1] iwork = \
        matlib.full( liwork, 0, dtype=ctypes.c_int )

    cdef lapack_int ret = solve_gep_with_deflation_double(
        n, &K[0,0], ldk, &M[0,0], ldm, &theta[0], &rank_M,
        &work[0], lwork, &iwork[0], liwork)

    return ret, K, theta



def deflation_workspace_query(lapack_int n):
    assert n > 0

    cdef double opt = N.nan
    cdef lapack_int iopt = -1

    cdef lapack_int ret = solve_gep_with_deflation_double( \
        n, NULL, n, NULL, n, NULL, NULL, &opt, -1, &iopt, -1)
    assert ret == 0

    assert opt > 0
    assert round(opt) == opt

    return int(opt), iopt
