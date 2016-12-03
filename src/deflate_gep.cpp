/*
 * Copyright 2015 Christoph Conrads
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * This Source Code Form is "Incompatible With Secondary Licenses", as
 * defined by the Mozilla Public License, v. 2.0.
 */
#include <blas.hpp>
#include <lapack.hpp>
#include <matrix.hpp>

#include <limits>
#include <cmath>

#include <algorithm>
#include <functional>

#include <cassert>



typedef lapack::integer_t integer_t;



/**
 * This function deflates infinite eigenvalues from a Hermitian matrix pencil.
 *
 * The function employs Mehrmann/Xu deflation.
 *
 * @param[in,out] K On entry, the upper triangle contains the stiffness matrix;
 *                  on exit, the matrix A if M is singular
 * @param[in,out] M On entry, the upper triangle contains the mass matrix;
 *                  on exit, the matrix B if M is singular
 * @param[out] rank_M The numerical rank of M
 * @param[out] X The matrix of eigenvectors of the unaltered M
 * @param[out] U If M is singular, the last rank_M columns contain a basis of
 *               the space of finite eigenvalues
 */
template<typename float_t>
integer_t deflate_gep_impl(
	integer_t n,
	float_t* K, integer_t ldk,
	float_t* M, integer_t ldm,
	float_t* lambda, integer_t* p_rank_M,
	float_t* X, integer_t ldx,
	float_t* U, integer_t ldu,
	float_t* workspace, integer_t workspace_size,
	integer_t* iwork, integer_t liwork)
{
	const integer_t minimum_workspace_size = 2*n*n + 6*n + 1;
	const integer_t minimum_iwork_size = 5*n + 3;


	// test for quick return
	if(n == 0)
		return 0;


	// check n
	if(n < 0)
		return -1;
	if(workspace == nullptr)
		return -11;
	if(iwork == nullptr)
		return -13;


	// workspace query
	if(workspace_size == -1)
	{
		float_t syevd_opt = -1;
		integer_t iwork_opt = -1;
		const integer_t syevd_ret = lapack::heevd(
			'V', 'U', n, nullptr, n, nullptr, &syevd_opt,-1, &iwork_opt,liwork);
		assert( syevd_ret == 0 );
		use(syevd_ret);

		float_t gesvd_opt = -1;
		const integer_t gesvd_ret = lapack::gesvd(
			'A', 'N', n, n, nullptr, n, nullptr, nullptr, n, nullptr, n,
			&gesvd_opt, -1);
		assert( gesvd_ret == 0 );
		use(gesvd_ret);

		workspace[0] = std::max(syevd_opt, gesvd_opt);
		iwork[0] = iwork_opt;
		return 0;
	}


	// check other parameters
	if(K == nullptr)
		return -2;
	if(ldk < n)
		return -3;
	if(M == nullptr)
		return -4;
	if(ldm < n)
		return -5;
	if(p_rank_M == nullptr)
		return -6;
	if(X == nullptr)
		return -7;
	if(ldx < n)
		return -8;
	if(U == nullptr)
		return -9;
	if(ldu < n)
		return -10;
	if(workspace_size < minimum_workspace_size)
		return -12;
	if( liwork < 1 || (n > 1 && liwork < minimum_iwork_size) )
		return -14;


	// introduce variables
	const float_t epsilon = std::numeric_limits<float_t>::epsilon();

#ifndef NDEBUG
	const float_t NaN = std::numeric_limits<float_t>::quiet_NaN();

	*p_rank_M = -1;
	lapack::laset('A', n, n, NaN, NaN, X, ldx);
	lapack::laset('A', n, n, NaN, NaN, U, ldu);
	std::fill_n(workspace, workspace_size, NaN);
	std::fill_n(iwork, liwork, -1);
#endif


	const float_t norm_K_frob = lapack::lanhe('F', 'U', n, K, ldk, nullptr);
	assert( !std::isnan(norm_K_frob) );


	// eigendecomposition M
	{
		const char up = 'U';
		lapack::lacpy(up, n, n, M, ldm, X, ldx);

		const char jobz = 'V';

		float_t* const work = workspace;
		const integer_t lwork = workspace_size;

		integer_t ret = lapack::heevd(
			jobz, up, n, X, ldx, lambda, work, lwork, iwork, liwork);
		assert( ret >= 0 );

		if( ret > 0 )
			return 1;

#ifndef NDEBUG
		// test eigenvalues
		bool (*isNaN)(float_t) = &std::isnan;
		assert( std::none_of(lambda, lambda+n, std::ref(isNaN)) );

		// eigenvalues should be in ascending order
		for(integer_t i = 0; i < n-1; ++i)
			assert( lambda[i] <= lambda[i+1] );
#endif
	}


	// determine mass matrix rank
	{
		const float_t lambda_max = lambda[n-1];


		const auto is_numerically_nonzero =
			[n, epsilon, lambda_max](float_t x) -> bool
			{ return std::abs(x) > n * epsilon * lambda_max; };

		const auto is_numerically_negative =
			[is_numerically_nonzero](float_t x) -> bool
			{return is_numerically_nonzero(x) && x < 0;};


		// mass matrix is not positive semidefinite
		if( std::any_of(lambda, lambda+n, is_numerically_negative) )
			return 2;


		*p_rank_M = std::count_if(lambda, lambda+n, is_numerically_nonzero);
	}


	// terminate if mass matrix has full rank
	if(*p_rank_M == n)
		return 0;


	// terminate if mass matrix is the matrix of zeros
	if(*p_rank_M == 0)
		return 0;


	const integer_t p = n - *p_rank_M;
	assert( p > 0 );
	assert( p < n );

	const integer_t r = *p_rank_M;


	// compute X^* K X
	{
		const char side = 'L';
		const char up = 'U';

		blas::hemm(side, up, n, n, 1.0, K, ldk, X, ldx, 0.0, M, ldm);

		const char transa = 'T';
		const char transb = 'N';

		blas::gemm(transa, transb, n, n, n, 1.0, X, ldx, M, ldm, 0.0, K, ldk);
	}


	// SVD K_M(:,1:p)
	{
		const char uplo = 'A';

		lapack::lacpy(uplo, n, p, K, ldk, M, ldm);


		const char jobu = 'A';
		const char jobvt = 'N';

		float_t* const sigma = workspace;

		float_t* const Vt = nullptr;
		const integer_t ldvt = n;

		float_t* const work = workspace + n;
		const integer_t lwork = workspace_size - n;

		const integer_t ret = lapack::gesvd(
			jobu, jobvt, n, p, M, ldm, sigma, U, ldu, Vt, ldvt, work, lwork);
		assert( ret >= 0 );

		if( ret > 0 )
			return 3;


		// test for nonregular matrix pencils
		const float_t sigma_min = sigma[p-1];
		if( sigma_min <= 2 * norm_K_frob * std::sqrt(n) * epsilon )
			return 4;
	}


	const float_t* const U22 = U + compute_offset(n, ldu, p+1, p+1);


	// compute A
	{
		const char side = 'L';
		const char up = 'U';
		const float_t* const K22 = K + compute_offset(n, ldk, p+1, p+1);

		blas::hemm(side, up, r, r, 1.0, K22, ldk, U22, ldu, 0.0, M, ldm);

		const char transb = 'N';

		const float_t* const K21 = K + compute_offset(n, ldk, p+1, 1);
		const float_t* const U12 = U + compute_offset(n, ldu, 1, p+1);

		blas::gemm('N', transb, r, r, p, 1.0, K21, ldk, U12, ldu, 1.0, M, ldm);

		blas::gemm('T', transb, r, r, r, 1.0, U22, ldu, M, ldm, 0.0, K, ldk);
	}


	// compute B
	{
		const char uplo = 'A';
		float_t* const D = lambda + p;
		float_t* const work = workspace;

		lapack::lacpy(uplo, r, r, U22, ldu, work, r);
		lapack::lascl2(r, r, D, work, r);

		blas::gemm('T', 'N', r, r, r, 1.0, U22, ldu, work, r, 0.0, M, ldm);
	}

	return 0;
}



integer_t deflate_gep(
	integer_t n,
	float* K, integer_t ldk,
	float* M, integer_t ldm,
	float* lambda, integer_t* p_rank_M,
	float* X, integer_t ldx,
	float* U, integer_t ldu,
	float* work, integer_t lwork,
	integer_t* iwork, integer_t liwork)
{
	return deflate_gep_impl(
		n, K, ldk, M, ldm, lambda, p_rank_M, X, ldx, U, ldu,
		work, lwork, iwork, liwork
	);
}



integer_t deflate_gep(
	integer_t n,
	double* K, integer_t ldk,
	double* M, integer_t ldm,
	double* lambda, integer_t* p_rank_M,
	double* X, integer_t ldx,
	double* U, integer_t ldu,
	double* work, integer_t lwork,
	integer_t* iwork, integer_t liwork)
{
	return deflate_gep_impl(
		n, K, ldk, M, ldm, lambda, p_rank_M, X, ldx, U, ldu,
		work, lwork, iwork, liwork);
}



extern "C" lapack_int deflate_gep_single(
	lapack_int n,
	float* K, lapack_int ldk,
	float* M, lapack_int ldm,
	float* lambda, lapack_int* p_rank_M,
	float* X, lapack_int ldx,
	float* U, lapack_int ldu,
	float* work, lapack_int lwork,
	lapack_int* iwork, lapack_int liwork)
{
	return deflate_gep_impl(
		n, K, ldk, M, ldm, lambda, p_rank_M, X, ldx, U, ldu,
		work, lwork, iwork, liwork);
}



extern "C" lapack_int deflate_gep_double(
	lapack_int n,
	double* K, lapack_int ldk,
	double* M, lapack_int ldm,
	double* lambda, lapack_int* p_rank_M,
	double* X, lapack_int ldx,
	double* U, lapack_int ldu,
	double* work, lapack_int lwork,
	lapack_int* iwork, lapack_int liwork)
{
	return deflate_gep_impl(
		n, K, ldk, M, ldm, lambda, p_rank_M, X, ldx, U, ldu,
		work, lwork, iwork, liwork);
}
