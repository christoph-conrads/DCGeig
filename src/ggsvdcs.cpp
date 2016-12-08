/*
 * Copyright 2015-2016 Christoph Conrads
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <ggsvdcs.hpp>
#include <blas.hpp>
#include <lapack.hpp>
#include <matrix.hpp>

#include <algorithm>

#include <use.hpp>
#include <cassert>


typedef lapack::integer_t Integer;


template<typename Number, typename Real>
Integer ggsvdcs_impl(
	char jobu1, char jobu2, char jobqt,
	Integer m, Integer n, Integer p, Integer* p_rank,
	Number* A, Integer lda, Number* B, Integer ldb, Real* theta,
	Number* U1, Integer ldu1, Number* U2, Integer ldu2,
	Number* Qt, Integer ldqt,
	Real* workspace, Integer workspace_size,
	Integer* iwork, Integer liwork)
{
	const Integer z = sizeof(Number) / sizeof(Real);
	assert( sizeof(Number) == z * sizeof(Real) );

	const Integer t = m+p - std::min({m, p, n, std::max(m+p-n,0)});

	const bool wantu = jobu1 == 'Y';
	const bool wantv = jobu2 == 'Y';
	const bool wantqt = jobqt == 'Y';

	// check input
	if(m < 0)
		return -4;
	if(n < 0)
		return -5;
	if(p < 0)
		return -6;
	if(!p_rank && workspace_size != -1)
		return -7;
	if(!A && workspace_size != -1)
		return -8;
	if(lda < std::max(1,m))
		return -9;
	if(!B && workspace_size != -1)
		return -10;
	if(ldb < std::max(1,p))
		return -11;
	if(!theta && workspace_size != -1)
		return -12;
	if( !U1 && wantu && workspace_size != -1 )
		return -13;
	if( wantu ? (ldu1 < std::max(1,m)) : (ldu1 < 1) )
		return -14;
	if( !U2 && wantv && workspace_size != -1 )
		return -15;
	if( wantv ? (ldu2 < std::max(1,p)) : (ldu2 < 1) )
		return -16;
	if( !Qt && wantqt && workspace_size != -1 )
		return -17;
	if( wantqt ? (ldqt < std::max(1,n)) : (ldqt < 1) )
		return -18;
	if(!workspace)
		return -19;
	if(workspace_size < 1 && workspace_size != -1)
		return -20;
	if(!iwork && workspace_size != -1)
		return -21;
	if(liwork < n + t && workspace_size != -1)
		return -22;


	// quick return?
	if( n == 0 || (m==0 && p==0) )
		return 0;


	// compute optimal workspace size?
	if(workspace_size == -1)
	{
		const Integer k = m+p;

		Real geqp3_opt = -1;
		Real ungqr_opt = -1;
		Real uncsd2by1_opt = -1;

		Integer info = -1;
		use(info);

		info = lapack::geqp3(
			k, n, nullptr, k, nullptr, nullptr, &geqp3_opt, -1);
		assert( info == 0 );
		assert( geqp3_opt >= n );

		info = lapack::ungqr(k, n, n, nullptr, k, nullptr, &ungqr_opt, -1);
		assert( info == 0 );
		assert( ungqr_opt >= n );

		info = lapack::uncsd2by1(
			jobu1, jobu2, jobqt, k, m, n, nullptr, m, nullptr, p, nullptr,
			nullptr, m, nullptr, p, nullptr, n, &uncsd2by1_opt, -1, nullptr);
		assert( info == 0 );
		assert( uncsd2by1_opt >= t );

		workspace[0] = k*n + std::max({ geqp3_opt, ungqr_opt, uncsd2by1_opt });

		return 0;
	}


	// initialize variables
	const Integer k = m+p;
	const Integer l = std::min(k,n);

	const Real eps = std::numeric_limits<Real>::epsilon();

	Number* const C = reinterpret_cast<Number*>(workspace);
	const Integer ldc = k;

	Real* const work = workspace + z * k*n;
	const Integer lwork = workspace_size - z * k*n;


#ifndef NDEBUG
	const Number NaN = std::numeric_limits<Number>::quiet_NaN();
	const Real NaR = std::numeric_limits<Real>::quiet_NaN();

	*p_rank = -1;

	lapack::laset('A', k, n, NaN, NaN, C, ldc);
	if(wantu) lapack::laset('A', m, m, NaN, NaN, U1, ldu1);
	if(wantv) lapack::laset('A', p, p, NaN, NaN, U2, ldu2);
	if(wantqt) lapack::laset('A', n, n, NaN, NaN, Qt, ldqt);

	std::fill_n(workspace, workspace_size, NaR);
	std::fill_n(iwork, liwork, -1);
#endif


	// copy matrices
	{
		lapack::lacpy('A', m, n, A, lda, C+0, ldc);
		lapack::lacpy('A', p, n, B, ldb, C+m, ldc);

#ifndef NDEBUG
		lapack::laset('A', m, n, NaN, NaN, A, lda);
		lapack::laset('A', p, n, NaN, NaN, B, ldb);
#endif
	}


	const Real normF_C = lapack::lange('F', k, n, C, ldc);


	// compute thin QR factorization with column pivoting
	Integer* const pivot = iwork;
	Number* const tau = theta;
	{
		assert( lwork >= 2*n + z * (n+1) );

		std::fill_n(pivot, n, 0);

		Integer info = -1;
		use(info);

		info = lapack::geqp3(k, n, C, ldc, pivot, tau, work, lwork);
		assert( info == 0 );
	}


	// compute rank
	{
		Integer r = 0;
		for(r = 0; r < l; ++r)
		{
			const Number* R_ii = C + compute_offset(l, n, ldc, r+1, r+1);

			if(std::abs(*R_ii) <= n * eps * normF_C)
				break;
		}

		*p_rank = r;
	}


	const Integer r = *p_rank;


	// copy R(QR)
	{
		assert( p >= r );
		lapack::lacpy('U', r, n, C, ldc, B, ldb);

#ifndef NDEBUG
		lapack::laset('U', l, n, NaN, NaN, C, ldc);
#endif
	}


	// explicitly compute Q(QR)
	{
		Integer info = -1;
		use(info);

		info = lapack::ungqr(k, r, r, C, ldc, tau, work, lwork);
		assert( info == 0 );
	}


	// compute CS decomposition
	{
		assert( liwork >= n + k - std::min({m,p,r,k-r}) );

		Number* const X11 = C + 0;
		Number* const X21 = C + m;

		const Integer info = lapack::uncsd2by1(
			jobu1, jobu2, jobqt, k, m, r,
			X11, ldc, X21, ldc, theta,
			U1, ldu1, U2, ldu2, Qt, ldqt,
			work, lwork, iwork+n);
		assert( info >= 0 );

		if(info > 0)
			return 1;
	}


#ifndef NDEBUG
	lapack::laset('A', k, n, NaN, NaN, C, ldc);
#endif


	// compute Q*, V(CSD), R(QR)
	// TODO: n =/= m, n =/= p, m =/= p
	{
		// restrict to equal dimensions
		assert( n == m );
		assert( n == p );

		// compute V*(CSD) R(QR)
		lapack::lacpy('A', r, r, Qt, ldqt, C, r);

		assert( r <= m );
		lapack::laset('L', r-1, n, 0.0, 0.0, B+1, ldb);
		blas::gemm('N', 'N', r, n, r, 1, C, r, B, ldb, 0.0, Qt+n-r, ldqt);

		// compute RQ decomposition
		Number* const tau = C;

		Integer info = -1;
		use(info);

		info = lapack::gerqf(r, n, Qt+n-r, ldqt, tau, work, lwork);
		assert( info == 0 );

		// copy R
		Number* const R = Qt + compute_offset(n, ldqt, n-r+1, n-r+1);
		lapack::lacpy('U', r, r, R, ldqt, A, lda);
#ifndef NDEBUG
		lapack::laset('U', r, r, NaN, NaN, R, ldqt);
#endif

		// create Q*
		info = lapack::ungrq(n, n, r, Qt, ldqt, tau, work, lwork);
		assert( info == 0 );

		// reverse permutation
		lapack::lapmt(false, n, n, Qt, ldqt, pivot);
	}

	return 0;
}




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
	lapack::integer_t* iwork, lapack::integer_t liwork)
{
	return ggsvdcs_impl(
		jobu1, jobu2, jobq, m, n, p, p_rank,
		A, lda, B, ldb, theta,
		U1, ldu1, U2, ldu2, Qt, ldqt,
		workspace, workspace_size, iwork, liwork);
}



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
	lapack::integer_t* iwork, lapack::integer_t liwork)
{
	return ggsvdcs_impl(
		jobu1, jobu2, jobq, m, n, p, p_rank,
		A, lda, B, ldb, theta,
		U1, ldu1, U2, ldu2, Qt, ldqt,
		workspace, workspace_size, iwork, liwork);
}
