/*
 * Copyright 2015-2016 Christoph Conrads
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * This Source Code Form is "Incompatible With Secondary Licenses", as
 * defined by the Mozilla Public License, v. 2.0.
 */
#pragma once

#include <config.h>

#ifdef USE_MKL
# include <mkl.h>
#else
# include <lapacke.h>
#endif

#include <stddef.h>


lapack_int solve_gep_with_qr_csd_single(
	lapack_int n,
	float* K, lapack_int ldk,
	float* M, lapack_int ldm,
	float* lambda, lapack_int* rank,
	float* work, lapack_int lwork,
	lapack_int* iwork, lapack_int liwork);

lapack_int solve_gep_with_qr_csd_double(
	lapack_int n,
	double* K, lapack_int ldk,
	double* M, lapack_int ldm,
	double* lambda, lapack_int* rank,
	double* work, lapack_int lwork,
	lapack_int* iwork, lapack_int liwork);



lapack_int solve_gep_with_gsvd_single(
	lapack_int n,
	float* K, lapack_int ldk,
	float* M, lapack_int ldm,
	float* lambda, lapack_int* rank,
	float* workspace, lapack_int workspace_size,
	lapack_int* iwork, lapack_int liwork);

lapack_int solve_gep_with_gsvd_double(
	lapack_int n,
	double* K, lapack_int ldk,
	double* M, lapack_int ldm,
	double* lambda, lapack_int* rank,
	double* workspace, lapack_int workspace_size,
	lapack_int* iwork, lapack_int liwork);



lapack_int deflate_gep_single(
	lapack_int n,
	float* K, lapack_int ldk,
	float* M, lapack_int ldm,
	float* lambda, lapack_int* rank_M,
	float* X, lapack_int ldx,
	float* Q, lapack_int ldq,
	float* work, lapack_int lwork,
	lapack_int* iwork, lapack_int liwork);

lapack_int deflate_gep_double(
	lapack_int n,
	double* K, lapack_int ldk,
	double* M, lapack_int ldm,
	double* lambda, lapack_int* rank_M,
	double* X, lapack_int ldx,
	double* Q, lapack_int ldq,
	double* work, lapack_int lwork,
	lapack_int* iwork, lapack_int liwork);



lapack_int solve_gep_with_deflation_single(
	lapack_int n,
	float* K, lapack_int ldk,
	float* M, lapack_int ldm,
	float* lambda, lapack_int* p_rank_M,
	float* work, lapack_int lwork,
	lapack_int* iwork, lapack_int liwork);

lapack_int solve_gep_with_deflation_double(
	lapack_int n,
	double* K, lapack_int ldk,
	double* M, lapack_int ldm,
	double* lambda, lapack_int* p_rank_M,
	double* work, lapack_int lwork,
	lapack_int* iwork, lapack_int liwork);
