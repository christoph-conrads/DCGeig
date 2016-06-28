/*
 * Copyright 2015-2016 Christoph Conrads
 *
 * This file is part of DCGeig and it is subject to the terms of the DCGeig
 * license. See http://DCGeig.tech/license for a copy of this license.
 */
#include <config.h>
#include <hpsd_gep_solvers.hpp>
#include <blas.hpp>
#include <matrix.hpp>

#include <cmath>
#include <limits>

#include <algorithm>

#include <cassert>



typedef lapack::integer_t integer_t;



template<typename float_t>
integer_t solve_gep_with_gsvd_impl(
	integer_t n,
	float_t* K, integer_t ldk,
	float_t* M, integer_t ldm,
	float_t* lambda, integer_t* p_rank,
	float_t* workspace, integer_t workspace_size,
	integer_t* iwork, integer_t liwork)
{
	const integer_t minimum_workspace_size = n*n + 6*n + 1;


	// test for quick return
	if(n == 0)
		return 0;


	// check n
	if(n < 0)
		return -1;
	if(workspace == nullptr)
		return -8;


	// compute optimal workspace size?
	if(workspace_size < 0)
	{
#ifdef HAS_GGSVD3
		float_t ggsvd3_opt = -1;

		integer_t ret = lapack::ggsvd3(
			'N', 'N', 'Q', n, n, n, nullptr, nullptr,
			nullptr, n, nullptr, n, nullptr, nullptr,
			nullptr, n, nullptr, n, nullptr, n,
			&ggsvd3_opt, -1, nullptr);
		assert( ret == 0 );
		use(ret);

		*workspace = n*n + 2*n + ggsvd3_opt;
#else
		workspace[0] = minimum_workspace_size;
#endif
		return 0;
	}


	// check other arguments
	if(K == nullptr)
		return -2;
	if(ldk < n)
		return -3;
	if(M == nullptr)
		return -4;
	if(ldm < n)
		return -5;
	if(lambda == nullptr)
		return -6;
	if(p_rank == nullptr)
		return -7;
	if(workspace_size < minimum_workspace_size && workspace_size != -1)
		return -9;
	if(iwork == nullptr)
		return -10;
	if(liwork < n)
		return -11;


	// initialize variables
	const float_t eps = std::numeric_limits<float_t>::epsilon();

	float_t* const C = workspace;
	const integer_t ldc = n;

	float_t* const alpha = C + n*n;
	float_t* const beta = alpha + n;

	float_t* const work = beta + n;
	const integer_t lwork = workspace_size - n*n - 2*n;


#ifndef NDEBUG
	const float_t NaN = std::numeric_limits<float_t>::quiet_NaN();

	*p_rank = -1;
	std::fill_n(lambda, n, NaN);
	std::fill_n(workspace, workspace_size, NaN);
	std::fill_n(iwork, liwork, -1);
#endif


	// factorize stiffness matrix
	{
		assert( lwork >= 2*n );
		assert( liwork >= n );

		const float_t normF_K = lapack::lanhe('F', 'U', n, K, ldk);
		assert( !std::isnan(normF_K) );

		lapack::lacpy('U', n, n, K, ldk, C, ldc);

		// Cholesky
		integer_t rank_K = -1;
		const float_t tol = 0;
		integer_t ret = -1;
		use(ret);

		integer_t* const pivot = iwork;

		ret = call_xpstrf('U', n, C, ldc, pivot, &rank_K, tol, work);
		assert( ret == 0 );

		// check if not positive semidefinite
		if(rank_K < n)
		{
			blas::gemm('T', 'N', n, n, rank_K, 1, C, ldc, C, ldc, -1, K, ldk);

			const float_t normF_K0 = lapack::lanhe('F', 'U', n, K, ldk);
			assert( !std::isnan(normF_K0) );

			if(normF_K0 > rank_K * eps * normF_K)
				return 1;
		}
	}


	// factorize mass matrix
	{
		assert( lwork >= 2*n );
		assert( liwork >= n );

		const float_t normF_M = lapack::lanhe('F', 'U', n, M, ldm);
		assert( !std::isnan(normF_M) );

		lapack::lacpy('U', n, n, M, ldm, K, ldk);

		// Cholesky
		integer_t rank_M = -1;
		const float_t tol = 0;
		integer_t ret = -1;
		use(ret);

		integer_t* const pivot = iwork;

		ret = call_xpstrf('U', n, M, ldm, pivot, &rank_M, tol, work);
		assert( ret == 0 );

		// check if not positive semidefinite
		if(rank_M < n)
		{
			blas::gemm('T', 'N', n, n, rank_M, 1, M, ldm, M, ldm, -1, K, ldk);

			const float_t normF_M0 = lapack::lanhe('F', 'U', n, K, ldk);
			assert( !std::isnan(normF_M0) );

			if(normF_M0 > rank_M * eps * normF_M)
				return 2;
		}
	}


	// direct GSVD
	integer_t k = -1;
	integer_t l = -1;
	{
		assert( liwork >= n );

		char jobu = 'N';
		char jobv = 'N';
		char jobq = 'Q';

		float_t* U = nullptr;
		integer_t ldu = n;
		float_t* V = nullptr;
		integer_t ldv = n;
		float_t* Q = K;
		integer_t ldq = ldk;

		integer_t ret = lapack::ggsvd3(
			jobu, jobv, jobq, n, n, n, &k, &l,
			C, ldc, M, ldm, alpha, beta, U, ldu, V, ldv, Q, ldq,
			work, lwork, iwork);

		assert( ret >= 0 );
		assert( k >= 0 );
		assert( k <= n );
		assert( l >= 0 );
		assert( l <= n );
		assert( k+l <= n );

		if(ret == 1)
			return 3;
	}


	*p_rank = k + l;


	const integer_t p = n - (l+k);
	assert( p >= 0 );
	assert( p <= n );

	float_t* const R = C + compute_offset(n, ldc, 1, p+1);
	const integer_t ldr = ldc;


	// compute eigenvectors
	{
		// Q_1p1 := Q(:, p+1:n)
		float_t* const Q_1p1 = K + compute_offset(n, ldk, 1, p+1);
		const integer_t ldq = ldk;

		blas::trsm('R', 'U', 'N', 'N', n, l+k, 1.0, R, ldr, Q_1p1, ldq);
	}


	// compute eigenvalues
	{
		const float_t infinity = std::numeric_limits<float_t>::infinity();

		std::fill(lambda, lambda+p, -1);
		std::fill(lambda+p, lambda+p+k, infinity);

		for(integer_t i=0; i < l; ++i)
		{
			float_t c = alpha[k+i];
			float_t s = beta[k+i];

			assert( !std::isnan(c) );
			assert( !std::isnan(s) );
			assert( std::abs(c*c + s*s - 1) <= 2.5 * eps );

			lambda[p+k+i] = std::pow(c/s, 2);
		}
	}

	return 0;
}



lapack::integer_t solve_gep_with_gsvd(
	lapack::integer_t n,
	float* K, lapack::integer_t ldk,
	float* M, lapack::integer_t ldm,
	float* lambda, lapack::integer_t* rank,
	float* workspace, lapack::integer_t workspace_size,
	lapack::integer_t* iwork, lapack::integer_t liwork)
{
	return solve_gep_with_gsvd_impl(
		n, K, ldk, M, ldm, lambda, rank,
		workspace, workspace_size, iwork, liwork);
}



lapack::integer_t solve_gep_with_gsvd(
	lapack::integer_t n,
	double* K, lapack::integer_t ldk,
	double* M, lapack::integer_t ldm,
	double* lambda, lapack::integer_t* rank,
	double* workspace, lapack::integer_t workspace_size,
	lapack::integer_t* iwork, lapack::integer_t liwork)
{
	return solve_gep_with_gsvd_impl(
		n, K, ldk, M, ldm, lambda, rank,
		workspace, workspace_size, iwork, liwork);
}



extern "C" lapack_int solve_gep_with_gsvd_single(
	lapack_int n,
	float* K, lapack_int ldk,
	float* M, lapack_int ldm,
	float* lambda, lapack_int* rank,
	float* workspace, lapack_int workspace_size,
	lapack_int* iwork, lapack_int liwork)
{
	return solve_gep_with_gsvd(
		n, K, ldk, M, ldm, lambda, rank,
		workspace, workspace_size, iwork, liwork);
}



extern "C" lapack_int solve_gep_with_gsvd_double(
	lapack_int n,
	double* K, lapack_int ldk,
	double* M, lapack_int ldm,
	double* lambda, lapack_int* rank,
	double* workspace, lapack_int workspace_size,
	lapack_int* iwork, lapack_int liwork)
{
	return solve_gep_with_gsvd(
		n, K, ldk, M, ldm, lambda, rank,
		workspace, workspace_size, iwork, liwork);
}
