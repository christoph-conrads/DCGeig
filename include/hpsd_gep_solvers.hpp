/*
 * Copyright 2016 Christoph Conrads
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * This Source Code Form is "Incompatible With Secondary Licenses", as
 * defined by the Mozilla Public License, v. 2.0.
 */
#pragma once

#include <cstddef>
#include <lapack.hpp>


lapack::integer_t solve_gep_with_qr_csd(
	lapack::integer_t n,
	float* K, lapack::integer_t ldk,
	float* M, lapack::integer_t ldm,
	float* lambda, lapack::integer_t* rank,
	float* work, lapack::integer_t lwork,
	lapack::integer_t* iwork, lapack::integer_t liwork);

lapack::integer_t solve_gep_with_qr_csd(
	lapack::integer_t n,
	double* K, lapack::integer_t ldk,
	double* M, lapack::integer_t ldm,
	double* lambda, lapack::integer_t* rank,
	double* work, lapack::integer_t lwork,
	lapack::integer_t* iwork, lapack::integer_t liwork);



lapack::integer_t solve_gep_with_gsvd(
	lapack::integer_t n,
	float* K, lapack::integer_t ldk,
	float* M, lapack::integer_t ldm,
	float* lambda, lapack::integer_t* rank,
	float* workspace, lapack::integer_t workspace_size,
	lapack::integer_t* iwork, lapack::integer_t liwork);

lapack::integer_t solve_gep_with_gsvd(
	lapack::integer_t n,
	double* K, lapack::integer_t ldk,
	double* M, lapack::integer_t ldm,
	double* lambda, lapack::integer_t* rank,
	double* workspace, lapack::integer_t workspace_size,
	lapack::integer_t* iwork, lapack::integer_t liwork);



lapack::integer_t deflate_gep(
	lapack::integer_t n,
	float* K, lapack::integer_t ldk,
	float* M, lapack::integer_t ldm,
	float* lambda, lapack::integer_t* rank_M,
	float* X, lapack::integer_t ldx,
	float* Q, lapack::integer_t ldq,
	float* work, lapack::integer_t lwork,
	lapack::integer_t* iwork, lapack::integer_t liwork);

lapack::integer_t deflate_gep(
	lapack::integer_t n,
	double* K, lapack::integer_t ldk,
	double* M, lapack::integer_t ldm,
	double* lambda, lapack::integer_t* rank_M,
	double* X, lapack::integer_t ldx,
	double* Q, lapack::integer_t ldq,
	double* work, lapack::integer_t lwork,
	lapack::integer_t* iwork, lapack::integer_t liwork);



lapack::integer_t solve_gep_with_deflation(
	lapack::integer_t n,
	float* K, lapack::integer_t ldk,
	float* M, lapack::integer_t ldm,
	float* lambda, lapack::integer_t* p_rank_M,
	float* work, lapack::integer_t lwork,
	lapack::integer_t* iwork, lapack::integer_t liwork);

lapack::integer_t solve_gep_with_deflation(
	lapack::integer_t n,
	double* K, lapack::integer_t ldk,
	double* M, lapack::integer_t ldm,
	double* lambda, lapack::integer_t* p_rank_M,
	double* work, lapack::integer_t lwork,
	lapack::integer_t* iwork, lapack::integer_t liwork);
