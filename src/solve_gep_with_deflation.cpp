/*
 * Copyright 2015-2016 Christoph Conrads
 *
 * This file is part of DCGeig and it is subject to the terms of the DCGeig
 * license. See http://DCGeig.tech/license for a copy of this license.
 */
#include <hpsd_gep_solvers.hpp>
extern "C" {
# include <hpsd_gep_solvers.h>
}

#include <blas.hpp>
#include <lapack.hpp>
#include <matrix.hpp>

#include <cmath>
#include <limits>

#include <algorithm>
#include <functional>

#include <cassert>



typedef lapack::integer_t integer_t;



template<typename float_t>
integer_t solve_gep_with_deflation_impl(
	integer_t n,
	float_t* K, integer_t ldk,
	float_t* M, integer_t ldm,
	float_t* lambda, integer_t* p_rank_M,
	float_t* workspace, integer_t workspace_size,
	integer_t* iwork, integer_t liwork)
{
	const integer_t minimum_lwork = 4*n*n + 6*n + 1;
	const integer_t minimum_liwork = 5*n + 3;


	// test for quick return
	if(n == 0)
		return 0;


	// check n
	if(n < 0)
		return -1;
	if(workspace == nullptr)
		return -8;
	if(iwork == nullptr)
		return -10;


	// workspace query?
	if(workspace_size == -1 || liwork == -1)
	{
		float_t syevd_opt = -1;
		integer_t syevd_iopt = -1;
		const integer_t syevd_ret = lapack::heevd(
			'V', 'U', n, nullptr, n, nullptr,
			&syevd_opt, -1, &syevd_iopt, -1);
		assert( syevd_ret == 0 );
		use(syevd_ret);

		float_t deflate_opt = -1;
		integer_t deflate_iopt = -1;
		const integer_t deflate_ret = deflate_gep(
			n, nullptr, n, nullptr, n, nullptr, nullptr,
			nullptr, n, nullptr, n,
			&deflate_opt, -1, &deflate_iopt, -1);
		assert( deflate_ret == 0 );
		use(deflate_ret);

		// special handing for n==1 required because Intel MKL 11.3.2 requires
		// *less* memory than LAPACK 3.6.0
		workspace[0] =
			(n==1) ? minimum_lwork : std::max(syevd_opt, deflate_opt) + 2*n*n;
		iwork[0] = std::max(syevd_iopt, deflate_iopt);
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
	if(p_rank_M == nullptr)
		return -7;
	if(workspace_size < minimum_lwork && workspace_size != -1)
		return -9;
	if( liwork < 1 || (n > 1 && liwork < minimum_liwork) )
		return -11;


	// initialize variables
	const float_t* const Y = workspace;
	const integer_t ldy = n;

	const float_t* const U = Y + n*n;
	const integer_t ldu = n;

	float_t* const work = workspace + 2*n*n;
	const integer_t lwork = workspace_size - 2*n*n;

#ifndef NDEBUG
	const float_t NaN = std::numeric_limits<float_t>::quiet_NaN();

	std::fill( lambda, lambda+n, NaN );
	std::fill_n( workspace, workspace_size, NaN );
#endif


	// deflate GEP
	{
		float_t* const Y = workspace;
		float_t* const U = Y + n*n;

		const integer_t ret = deflate_gep(
			n, K, ldk, M, ldm, lambda, p_rank_M, Y, ldy, U, ldu,
			work, lwork, iwork, liwork);
		assert( ret >= 0 );

		if( ret > 0 )
			return ret;

		assert( *p_rank_M <= n );
		assert( *p_rank_M >= 0 );


		if( *p_rank_M == 0 )
		{
			std::fill_n( lambda, n, std::numeric_limits<float_t>::infinity() );
			return 0;
		}


#ifndef NDEBUG
		bool (*isNaN)(double) = &std::isnan;

		assert( std::none_of(lambda, lambda+n, std::ref(isNaN)) );
		assert( std::none_of(Y, Y+n*n, std::ref(isNaN)) );
		if( *p_rank_M < n )
			assert( std::none_of(U, U+n*n, std::ref(isNaN)) );
#endif
	}


	const integer_t r = *p_rank_M;
	const integer_t p = n - r;


	// solve deflated GEP
	{
		const integer_t itype = 1;

		const integer_t ret = lapack::hegvd(
			itype, 'V', 'U', r, K, ldk, M, ldm, lambda+p,
			work, lwork, iwork, liwork);
		assert( ret >= 0 );

		if( ret > 0 && ret > r )
			return 5; // Cholesky decomposition incomplete
		if( ret > 0 )
			return 6; // no convergence
	}


	// revert basis changes
	if(r < n)
	{
		const float_t* const U1p = U + compute_offset(n, ldu, 1, p+1);
		float_t* const K1p = K + compute_offset(n, ldk, 1, p+1);

		blas::gemm('N', 'N', n, r, r, 1.0, U1p, ldu, K, ldk, 0.0, M, ldm);

		blas::gemm('N', 'N', n, r, n, 1.0, Y, ldy, M, ldm, 0.0, K1p, ldk);
		lapack::lacpy('A', n, p, Y, ldy, K, ldk);
	}


	// set infinite eigenvalues
	{
		const float_t inf = std::numeric_limits<float_t>::infinity();
		std::fill(lambda, lambda+p, inf);
	}


	return 0;
}



integer_t solve_gep_with_deflation(
	integer_t n,
	float* K, integer_t ldk,
	float* M, integer_t ldm,
	float* lambda, integer_t* p_rank_M,
	float* work, integer_t lwork,
	integer_t* iwork, integer_t liwork)
{
	return solve_gep_with_deflation_impl(
		n, K, ldk, M, ldm, lambda, p_rank_M, work, lwork, iwork, liwork);
}

integer_t solve_gep_with_deflation(
	integer_t n,
	double* K, integer_t ldk,
	double* M, integer_t ldm,
	double* lambda, integer_t* p_rank_M,
	double* work, integer_t lwork,
	integer_t* iwork, integer_t liwork)
{
	return solve_gep_with_deflation_impl(
		n, K, ldk, M, ldm, lambda, p_rank_M, work, lwork, iwork, liwork);
}



lapack_int solve_gep_with_deflation_single(
	lapack_int n,
	float* K, lapack_int ldk,
	float* M, lapack_int ldm,
	float* lambda, lapack_int* p_rank_M,
	float* work, lapack_int lwork,
	lapack_int* iwork, lapack_int liwork)
{
	return solve_gep_with_deflation_impl(
		n, K, ldk, M, ldm, lambda, p_rank_M, work, lwork, iwork, liwork);
}

lapack_int solve_gep_with_deflation_double(
	lapack_int n,
	double* K, lapack_int ldk,
	double* M, lapack_int ldm,
	double* lambda, lapack_int* p_rank_M,
	double* work, lapack_int lwork,
	lapack_int* iwork, lapack_int liwork)
{
	return solve_gep_with_deflation_impl(
		n, K, ldk, M, ldm, lambda, p_rank_M, work, lwork, iwork, liwork);
}
