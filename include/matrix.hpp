/*
 * Copyright 2015 Christoph Conrads
 *
 * This file is part of DCGeig and it is subject to the terms of the DCGeig
 * license. See https://DCGeig.tech/license for a copy of this license.
 */
#pragma once

#include <cstddef>
#include <use.hpp>

#include <algorithm>

#include "lapack.hpp"
#include <limits>

#include <cassert>


inline std::size_t compute_offset(
	std::size_t m, std::size_t n, std::size_t lda, std::size_t i, std::size_t j)
{
	use(m);
	use(n);

	assert( lda >= m );

	assert( i >= 1 );
	assert( i <= m );

	assert( j >= 1 );
	assert( j <= n );

	std::size_t offset = (j-1) * lda + i - 1;

	assert(offset < n*lda);

	return offset;
}



/**
 * This function computes offset for the entry (i,j) in an n by n square matrix
 * with stride lda and one-based indexing.
 *
 * @param[in] n The number of degrees of freedom
 * @param[in] lda The stride of the matrix
 */
inline std::size_t compute_offset(
	std::size_t n, std::size_t lda, std::size_t i, std::size_t j)
{
	return compute_offset(n, n, lda, i, j);
}



template<typename float_t>
lapack::integer_t call_xpstrf(
	const char uplo, lapack::integer_t n, float_t* A, lapack::integer_t lda,
	lapack::integer_t* pivot, lapack::integer_t* p_rank, float_t tol,
	float_t* work)
{
	typedef lapack::integer_t integer_t;

	if(uplo != 'U') // 'L' is not supported
		return -1;
	if(n < 0)
		return -2;
	if(A == nullptr)
		return -3;
	if(lda < n)
		return -4;
	if(pivot == nullptr)
		return -5;
	if(p_rank == nullptr)
		return -6;
	if(std::isnan(tol) || std::isinf(tol))
		return -7;
	if(work == nullptr)
		return -8;


#ifndef NDEBUG
	const float_t NaN = std::numeric_limits<float_t>::quiet_NaN();

	*p_rank = -1;
	std::fill_n(pivot, n, -1);
	std::fill_n(work, 2*n, NaN);
#endif


	if(n == 0)
	{
		*p_rank = 0;
		return 0;
	}


	// zero the lower triangular part
	lapack::laset('L', n-1, n, 0.0, 0.0, A+1, lda);


	// call xPSTRF
	{
		integer_t ret = -1;
		use(ret);

		ret = lapack::pstrf(uplo, n, A, lda, pivot, p_rank, tol, work);
		assert( ret >= 0 );
	}


	const integer_t r = *p_rank;
	const integer_t p = n - r;


	// deal with remainder
	if(p > 0)
	{
		float_t* const A_0 = A + compute_offset(n, lda, r+1, r+1);

		lapack::laset(uplo, p, p, 0.0, 0.0, A_0, lda);
	}


	// permute columns
	if(r > 0)
	{
		const bool backward = false;
		lapack::lapmt(backward, r, n, A, lda, pivot);
	}

	return 0;
}
