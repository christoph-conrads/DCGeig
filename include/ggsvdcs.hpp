/*
 * Copyright 2015 Christoph Conrads
 *
 * This file is part of DCGeig and it is subject to the terms of the DCGeig
 * license. See https://DCGeig.tech/license for a copy of this license.
 */
#include <lapack.hpp>



lapack::integer_t ggsvdcs(
	char jobu1, char jobu2, char jobq,
	lapack::integer_t m, lapack::integer_t n, lapack::integer_t p,
	lapack::integer_t* p_rank,
	float* A, lapack::integer_t lda,
	float* B, lapack::integer_t ldb,
	float* theta,
	float* U1, lapack::integer_t ldu1,
	float* U2, lapack::integer_t ldu2,
	float* Qt, lapack::integer_t ldqt,
	float* workspace, lapack::integer_t workspace_size,
	lapack::integer_t* iwork, lapack::integer_t liwork);

lapack::integer_t ggsvdcs(
	char jobu1, char jobu2, char jobq,
	lapack::integer_t m, lapack::integer_t n, lapack::integer_t p,
	lapack::integer_t* p_rank,
	double* A, lapack::integer_t lda,
	double* B, lapack::integer_t ldb,
	double* theta,
	double* U1, lapack::integer_t ldu1,
	double* U2, lapack::integer_t ldu2,
	double* Qt, lapack::integer_t ldqt,
	double* workspace, lapack::integer_t workspace_size,
	lapack::integer_t* iwork, lapack::integer_t liwork);
