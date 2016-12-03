/*
 * Copyright 2015-2016 Christoph Conrads
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * This Source Code Form is "Incompatible With Secondary Licenses", as
 * defined by the Mozilla Public License, v. 2.0.
 */
#include <hpsd_gep_solvers.hpp>
#include <blas.hpp>
#include <matrix.hpp>
#include <ggsvdcs.hpp>

extern "C" {
# include <hpsd_gep_solvers.h>
}

#include <cmath>
#include <limits>

#include <algorithm>

#include <cassert>



typedef lapack::integer_t integer_t;



template<typename float_t>
integer_t solve_gep_with_qr_csd_impl(
	integer_t n,
	float_t* K, integer_t ldk,
	float_t* M, integer_t ldm,
	float_t* lambda, integer_t* p_rank,
	float_t* workspace, integer_t workspace_size,
	integer_t* iwork, integer_t liwork)
{
	// xUNCSD2BY1 minimum workspace size in LAPACK 3.5.0:
	// LWORKMIN = IBBCSD+LBBCSD-1 = MAX(11, 9*n-3) + 8*n - 1
	const integer_t minimum_workspace_size = 3*n*n + std::max(18, 17*n - 4);
	const integer_t minimum_liwork = 3*n;


	// check n
	if(n < 0)
		return -1;
	if(K == nullptr && workspace_size != -1)
		return -2;
	if(ldk < n)
		return -3;
	if(M == nullptr && workspace_size != -1)
		return -4;
	if(ldm < n)
		return -5;
	if(lambda == nullptr && workspace_size != -1)
		return -6;
	if(p_rank == nullptr && workspace_size != -1)
		return -7;
	if(workspace == nullptr)
		return -8;
	if(workspace_size < minimum_workspace_size && workspace_size != -1)
		return -9;
	if(iwork == nullptr && workspace_size != -1)
		return -10;
	if(liwork < minimum_liwork && workspace_size != -1)
		return -11;


	// test for quick return
	if(n == 0)
		return 0;


	// workspace query
	if(workspace_size == -1)
	{
		const float_t pstrf_opt = 2 * n;
		float_t ggsvdcs_opt = -1;

		const integer_t info = ggsvdcs(
			'N', 'N', 'Y', n, n, n, p_rank,
			nullptr, n, nullptr, n, nullptr,
			nullptr, n, nullptr, n, nullptr, n,
			&ggsvdcs_opt, -1, nullptr, -1);
		assert( info == 0 );
		use(info);

		workspace[0] = n*n + std::max({pstrf_opt, ggsvdcs_opt});

		return 0;
	}


	// initialize variables
	const float_t NaN = std::numeric_limits<float_t>::quiet_NaN();
	const float_t eps = std::numeric_limits<float_t>::epsilon();
	const float_t inf = std::numeric_limits<float_t>::infinity();


	float_t* const C = workspace;
	const integer_t ldc = n;

	float_t* const work = C + n*n;
	const integer_t lwork = workspace_size - n*n;

#ifndef NDEBUG
	lapack::laset('L', n-1, n, NaN, NaN, K+1, ldk);
	lapack::laset('L', n-1, n, NaN, NaN, M+1, ldm);
	std::fill_n(lambda, n, NaN);
	*p_rank = -1;
	std::fill_n(workspace, workspace_size, NaN);
	std::fill_n(iwork, liwork, -1);
#endif


	// scale mass matrix
	float_t s = NaN;
	{
		const float_t normF_K = lapack::lanhe('F', 'U', n, K, ldk);
		const float_t normF_M = lapack::lanhe('F', 'U', n, M, ldm);
		assert( !std::isnan(normF_K) );
		assert( !std::isnan(normF_M) );


		if(normF_K == 0 || normF_M == 0)
		{
			lapack::laset('A', n, n, 0.0, 1.0, K, ldk);
			if(normF_M == 0)
				std::fill_n(lambda, n, inf);
			else
				std::fill_n(lambda, n, 0);

			return 0;
		}

		const float_t t = std::log2(normF_K / normF_M);
		s = std::pow(2, std::round(t));

		const integer_t info =
			lapack::lascl('U', -1, -1, 1, s, n, n, M, ldm);
		assert( info == 0 );
		use(info);
	}


	// factorize mass matrix
	{
		assert( lwork >= 2*n );
		assert( liwork >= n );

		const float_t normF_M = lapack::lanhe('F', 'U', n, M, ldm);
		assert( !std::isnan(normF_M) );

		lapack::lacpy('U', n, n, M, ldm, C, ldc);

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
			blas::gemm('T', 'N', n, n, rank_M, 1, M, ldm, M, ldm, -1, C, ldc);

			const float_t normF_M0 = lapack::lanhe('F', 'U', n, C, ldc);
			assert( !std::isnan(normF_M0) );

			if(normF_M0 > rank_M * eps * normF_M)
				return 2;
		}
	}


#ifndef NDEBUG
	lapack::laset('A', n, n, NaN, NaN, C, ldc);
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


#ifndef NDEBUG
	lapack::laset('A', n, n, NaN, NaN, K, ldk);
#endif


	// compute GSVD
	{
		float_t* const U1 = nullptr;
		const integer_t ldu1 = n;

		float_t* const U2 = nullptr;
		const integer_t ldu2 = n;

		float_t* const Qt = K;
		const integer_t ldqt = ldk;

		integer_t info = ggsvdcs(
			'N', 'N', 'Y', n, n, n, p_rank,
			M, ldm, C, ldc, lambda,
			U1, ldu1, U2, ldu2, Qt, ldqt,
			work, lwork, iwork, liwork);
		assert( info >= 0 );

		if(info > 0)
			return 3;
	}


	const integer_t r = *p_rank;
	const integer_t p = n - r;


	// compute eigenvalues
	{
		std::copy_backward(lambda, lambda+r, lambda+n);

		std::fill(lambda, lambda+p, -1);

		auto fun =
			[s, eps, inf](float_t f) -> float_t
			{
				const float_t a = std::cos(f);
				const float_t b = std::sin(f);
				assert( std::abs(a*a + b*b - 1) <= 2.5 * eps );
				use(a);
				use(b);

				if(f >= M_PI_2)
					return inf;

				return s * std::pow(std::tan(f), 2);
			};

		std::transform(lambda+p, lambda+n, lambda+p, fun);
	}


	// compute eigenvectors: X^* = R^{-*} Q^*
	{
		const float_t* const R = M;
		const integer_t ldr = ldm;

		float_t* const Qt_ran = K + p;
		const integer_t ldqt = ldk;

		blas::trsm('L', 'U', 'C', 'N', r, n, 1.0, R, ldr, Qt_ran, ldqt);
	}

	return 0;
}



integer_t solve_gep_with_qr_csd(
	integer_t n,
	float* K, integer_t ldk,
	float* M, integer_t ldm,
	float* lambda, integer_t* rank,
	float* work, integer_t lwork,
	integer_t* iwork, integer_t liwork)
{
	return solve_gep_with_qr_csd_impl(
		n, K, ldk, M, ldm, lambda, rank, work, lwork, iwork, liwork);
}



integer_t solve_gep_with_qr_csd(
	integer_t n,
	double* K, integer_t ldk,
	double* M, integer_t ldm,
	double* lambda, integer_t* rank,
	double* work, integer_t lwork,
	integer_t* iwork, integer_t liwork)
{
	return solve_gep_with_qr_csd_impl(
		n, K, ldk, M, ldm, lambda, rank, work, lwork, iwork, liwork);
}


integer_t solve_gep_with_qr_csd_single(
	integer_t n,
	float* K, integer_t ldk,
	float* M, integer_t ldm,
	float* lambda, integer_t* rank,
	float* work, integer_t lwork,
	integer_t* iwork, integer_t liwork)
{
	return solve_gep_with_qr_csd_impl(
		n, K, ldk, M, ldm, lambda, rank, work, lwork, iwork, liwork);
}


integer_t solve_gep_with_qr_csd_double(
	integer_t n,
	double* K, integer_t ldk,
	double* M, integer_t ldm,
	double* lambda, integer_t* rank,
	double* work, integer_t lwork,
	integer_t* iwork, integer_t liwork)
{
	return solve_gep_with_qr_csd_impl(
		n, K, ldk, M, ldm, lambda, rank, work, lwork, iwork, liwork);
}
