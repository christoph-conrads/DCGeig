/*
 * Copyright 2015-2016 Christoph Conrads
 *
 * This file is part of DCGeig and it is subject to the terms of the DCGeig
 * license. See https://DCGeig.tech/license for a copy of this license.
 */
#pragma once

#include <config.h>

#if not defined HAS_GGSVD3
# include <algorithm>
#endif

#ifdef USE_MKL
# include <mkl.h>
#else
# include <lapacke.h>
#endif

#include <cctype>
#include <cassert>


// add declarations not present in LAPACK 3.5.0
extern "C"
{
void slapmt(
	const lapack_int* forward, const lapack_int* m, const lapack_int* n,
	float* A, const lapack_int* lda, lapack_int* piv);

void dlapmt(
	const lapack_int* forward, const lapack_int* m, const lapack_int* n,
	double* A, const lapack_int* lda, lapack_int* piv);



void slascl(
	const char* type, const lapack_int* kl, const lapack_int* ku,
	const float* cfrom, const float* cto,
	const lapack_int* m, const lapack_int* n,
	float* A, const lapack_int* lda, lapack_int* info);

void dlascl(
	const char* type, const lapack_int* kl, const lapack_int* ku,
	const double* cfrom, const double* cto,
	const lapack_int* m, const lapack_int* n,
	double* A, const lapack_int* lda, lapack_int* info);



void slascl2(
	const lapack_int* m, const lapack_int* n,
	const float* D, float* X, const lapack_int* ldx);

void dlascl2(
	const lapack_int* m, const lapack_int* n,
	const double* D, double* X, const lapack_int* ldx);



void sorcsd2by1(
	const char* jobu1, const char* jobu2, const char* jobv1t,
	const lapack_int* m, const lapack_int* p, const lapack_int* q,
	float* X11, const lapack_int* ldx11,
	float* X21, const lapack_int* ldx21,
	float* theta,
	float* U1, const lapack_int* ldu1,
	float* U2, const lapack_int* ldu2,
	float* V1t, const lapack_int* ldv1t,
	float* work, const lapack_int* lwork, lapack_int* iwork, lapack_int* info);

void dorcsd2by1(
	const char* jobu1, const char* jobu2, const char* jobv1t,
	const lapack_int* m, const lapack_int* p, const lapack_int* q,
	double* X11, const lapack_int* ldx11,
	double* X21, const lapack_int* ldx21,
	double* theta,
	double* U1, const lapack_int* ldu1,
	double* U2, const lapack_int* ldu2,
	double* V1t, const lapack_int* ldv1t,
	double* work, const lapack_int* lwork, lapack_int* iwork, lapack_int* info);
} // extern "C"



namespace lapack
{

typedef lapack_int integer_t;



inline integer_t geqp3(
	integer_t m, integer_t n, float* A, integer_t lda,
	integer_t* pivot, float* tau, float* work, integer_t lwork)
{
	integer_t info = -1;
	sgeqp3(&m, &n, A, &lda, pivot, tau, work, &lwork, &info);
	return info;
}

inline integer_t geqp3(
	integer_t m, integer_t n, double* A, integer_t lda,
	integer_t* pivot, double* tau, double* work, integer_t lwork)
{
	integer_t info = -1;
	dgeqp3(&m, &n, A, &lda, pivot, tau, work, &lwork, &info);
	return info;
}



inline integer_t geqrf(
	integer_t m, integer_t n, float* A, integer_t lda, float* p_tau,
	float* p_work, integer_t lwork)
{
	integer_t info = -1;
	sgeqrf(&m, &n, A, &lda, p_tau, p_work, &lwork, &info);
	return info;
}

inline integer_t geqrf(
	integer_t m, integer_t n, double* A, integer_t lda, double* p_tau,
	double* p_work, integer_t lwork)
{
	integer_t info = -1;
	dgeqrf(&m, &n, A, &lda, p_tau, p_work, &lwork, &info);
	return info;
}



inline integer_t gerqf(
	integer_t m, integer_t n, float* A, integer_t lda, float* p_tau,
	float* p_work, integer_t lwork)
{
	integer_t info = -1;
	sgerqf(&m, &n, A, &lda, p_tau, p_work, &lwork, &info);
	return info;
}

inline integer_t gerqf(
	integer_t m, integer_t n, double* A, integer_t lda, double* p_tau,
	double* p_work, integer_t lwork)
{
	integer_t info = -1;
	dgerqf(&m, &n, A, &lda, p_tau, p_work, &lwork, &info);
	return info;
}



inline integer_t gesvd(
	char jobu, char jobvt, integer_t m, integer_t n, float* A, integer_t lda,
	float* sigma, float* U, integer_t ldu, float* Vt, integer_t ldvt,
	float* work, integer_t lwork)
{
	integer_t info = -1;
	sgesvd(
		&jobu, &jobvt, &m, &n, A, &lda, sigma, U, &ldu, Vt, &ldvt,
		work, &lwork, &info);
	return info;
}

inline integer_t gesvd(
	char jobu, char jobvt, integer_t m, integer_t n, double* A, integer_t lda,
	double* sigma, double* U, integer_t ldu, double* Vt, integer_t ldvt,
	double* work, integer_t lwork)
{
	integer_t info = -1;
	dgesvd(
		&jobu, &jobvt, &m, &n, A, &lda, sigma, U, &ldu, Vt, &ldvt,
		work, &lwork, &info);
	return info;
}



inline integer_t ggsvd3(
	char jobu, char jobv, char jobq, integer_t m, integer_t n, integer_t p,
	integer_t* p_k, integer_t* p_l,
	float* A, integer_t lda, float* B, integer_t ldb,
	float* alpha, float* beta,
	float* U, integer_t ldu, float* V, integer_t ldv, float* Q, integer_t ldq,
	float* work, integer_t lwork, integer_t* iwork)
{
#ifdef HAS_GGSVD3
	integer_t info = -1;
	sggsvd3(
		&jobu, &jobv, &jobq, &m, &n, &p, p_k, p_l,
		A, &lda, B, &ldb, alpha, beta, U, &ldu, V, &ldv, Q, &ldq,
		work, &lwork, iwork, &info);
	return info;
#else
	const integer_t lwork_min = std::max({3*n, m, p}) + n;
	if(lwork == -1)
	{
		work[0] = lwork_min;
		return 0;
	}

	if(lwork < lwork_min)
		return -22;

	integer_t info = -1;
	sggsvd(
		&jobu, &jobv, &jobq, &m, &n, &p, p_k, p_l,
		A, &lda, B, &ldb, alpha, beta, U, &ldu, V, &ldv, Q, &ldq,
		work, iwork, &info);
	return info;
#endif
}

inline integer_t ggsvd3(
	char jobu, char jobv, char jobq, integer_t m, integer_t n, integer_t p,
	integer_t* p_k, integer_t* p_l,
	double* A, integer_t lda, double* B, integer_t ldb,
	double* alpha, double* beta,
	double* U, integer_t ldu, double* V, integer_t ldv, double* Q, integer_t ldq,
	double* work, integer_t lwork, integer_t* iwork)
{
#ifdef HAS_GGSVD3
	integer_t info = -1;
	dggsvd3(
		&jobu, &jobv, &jobq, &m, &n, &p, p_k, p_l,
		A, &lda, B, &ldb, alpha, beta, U, &ldu, V, &ldv, Q, &ldq,
		work, &lwork, iwork, &info);
	return info;
#else
	const integer_t lwork_min = std::max({3*n, m, p}) + n;
	if(lwork == -1)
	{
		work[0] = lwork_min;
		return 0;
	}

	if(lwork < lwork_min)
		return -22;

	integer_t info = -1;
	dggsvd(
		&jobu, &jobv, &jobq, &m, &n, &p, p_k, p_l,
		A, &lda, B, &ldb, alpha, beta, U, &ldu, V, &ldv, Q, &ldq,
		work, iwork, &info);
	return info;
#endif
}



inline integer_t heevd(
	char jobz, char uplo, integer_t n, float* A, integer_t lda, float* lambda,
	float* work, integer_t lwork, integer_t* iwork, integer_t liwork)
{
	integer_t info = -1;
	ssyevd(
		&jobz, &uplo, &n, A, &lda, lambda, work, &lwork, iwork, &liwork, &info);
	return info;
}

inline integer_t heevd(
	char jobz, char uplo, integer_t n, double* A, integer_t lda, double* lambda,
	double* work, integer_t lwork, integer_t* iwork, integer_t liwork)
{
	integer_t info = -1;
	dsyevd(
		&jobz, &uplo, &n, A, &lda, lambda, work, &lwork, iwork, &liwork, &info);
	return info;
}



inline integer_t hegv(
	integer_t itype, char jobz, char uplo, integer_t n,
	float* A, lapack_int lda, float* B, lapack_int ldb, float* lambda,
	float* work, integer_t lwork)
{
	integer_t info = -1;
	ssygv(
		&itype, &jobz, &uplo, &n, A, &lda, B, &ldb, lambda,
		work, &lwork, &info);
	return info;
}

inline integer_t hegv(
	integer_t itype, char jobz, char uplo, integer_t n,
	double* A, lapack_int lda, double* B, lapack_int ldb, double* lambda,
	double* work, integer_t lwork)
{
	integer_t info = -1;
	dsygv(
		&itype, &jobz, &uplo, &n, A, &lda, B, &ldb, lambda,
		work, &lwork, &info);
	return info;
}



inline integer_t hegvd(
	integer_t itype, char jobz, char uplo, integer_t n,
	float* A, lapack_int lda, float* B, lapack_int ldb, float* lambda,
	float* work, integer_t lwork, integer_t* iwork, integer_t liwork)
{
	integer_t info = -1;
	ssygvd(
		&itype, &jobz, &uplo, &n, A, &lda, B, &ldb, lambda,
		work, &lwork, iwork, &liwork, &info);
	return info;
}

inline integer_t hegvd(
	integer_t itype, char jobz, char uplo, integer_t n,
	double* A, lapack_int lda, double* B, lapack_int ldb, double* lambda,
	double* work, integer_t lwork, integer_t* iwork, integer_t liwork)
{
	integer_t info = -1;
	dsygvd(
		&itype, &jobz, &uplo, &n, A, &lda, B, &ldb, lambda,
		work, &lwork, iwork, &liwork, &info);
	return info;
}



inline void lacpy(
	char uplo, integer_t m, integer_t n,
	const float* A, integer_t lda, float* B, integer_t ldb )
{
	slacpy( &uplo, &m, &n, A, &lda, B, &ldb );
}

inline void lacpy(
	char uplo, integer_t m, integer_t n,
	const double* A, integer_t lda, double* B, integer_t ldb )
{
	dlacpy( &uplo, &m, &n, A, &lda, B, &ldb );
}



inline float lange(
	char norm, integer_t m, integer_t n,
	const float* A, integer_t lda, float* work=nullptr)
{
	assert( (std::toupper(norm) != 'I') || work );
	return slange(&norm, &m, &n, A, &lda, work);
}

inline double lange(
	char norm, integer_t m, integer_t n,
	const double* A, integer_t lda, double* work=nullptr)
{
	assert( (std::toupper(norm) != 'I') || work );
	return dlange(&norm, &m, &n, A, &lda, work);
}



inline float lanhe(
	char norm, char uplo, integer_t n,
	const float* A, integer_t lda, float* work=nullptr)
{
	assert( (std::toupper(norm) != 'I') || work );

#ifdef USE_MKL
	// work around a bug with Intel MKL 11.3 Update 2
	const int num_threads = MKL_Set_Num_Threads_Local(1);
	const float ret = slansy(&norm, &uplo, &n, A, &lda, work);
	MKL_Set_Num_Threads_Local(num_threads);
	return ret;
#else
	return slansy(&norm, &uplo, &n, A, &lda, work);
#endif
}

inline double lanhe(
	char norm, char uplo, integer_t n,
	const double* A, integer_t lda, double* work=nullptr)
{
	assert( (std::toupper(norm) != 'I') || work );

#ifdef USE_MKL
	// work around a bug with Intel MKL 11.3 Update 2
	const int num_threads = MKL_Set_Num_Threads_Local(1);
	const double ret = dlansy(&norm, &uplo, &n, A, &lda, work);
	MKL_Set_Num_Threads_Local(num_threads);
	return ret;
#else
	return dlansy(&norm, &uplo, &n, A, &lda, work);
#endif
}



inline void lapmt(
	bool forward, integer_t m, integer_t n,
	float* A, integer_t lda, integer_t* piv)
{
	integer_t iforward = forward;
	slapmt( &iforward, &m, &n, A, &lda, piv );
}

inline void lapmt(
	bool forward, integer_t m, integer_t n,
	double* A, integer_t lda, integer_t* piv)
{
	integer_t iforward = forward;
	dlapmt( &iforward, &m, &n, A, &lda, piv );
}



inline integer_t lascl(
	char type, integer_t kl, integer_t ku, float cfrom, float cto,
	integer_t m, integer_t n, float* A, integer_t lda)
{
	integer_t info = -1;
	slascl(&type, &kl, &ku, &cfrom, &cto, &m, &n, A, &lda, &info);
	return info;
}

inline integer_t lascl(
	char type, integer_t kl, integer_t ku, double cfrom, double cto,
	integer_t m, integer_t n, double* A, integer_t lda)
{
	integer_t info = -1;
	dlascl(&type, &kl, &ku, &cfrom, &cto, &m, &n, A, &lda, &info);
	return info;
}



inline void lascl2(
	integer_t m, integer_t n, const float* D, float* X, integer_t ldx)
{
	slascl2(&m, &n, D, X, &ldx);
}

inline void lascl2(
	integer_t m, integer_t n, const double* D, double* X, integer_t ldx)
{
	dlascl2(&m, &n, D, X, &ldx);
}



inline void laset(
	char uplo, integer_t m, integer_t n,
	float alpha, float beta, float* A, integer_t lda )
{
	slaset( &uplo, &m, &n, &alpha, &beta, A, &lda );
}

inline void laset(
	char uplo, integer_t m, integer_t n,
	double alpha, double beta, double* A, integer_t lda )
{
	dlaset( &uplo, &m, &n, &alpha, &beta, A, &lda );
}



inline integer_t pstrf(
	char uplo, integer_t n, float* A, integer_t lda,
	integer_t* p_piv, integer_t* p_rank, float tol,
	float* p_work)
{
	integer_t info = -1;
	spstrf( &uplo, &n, A, &lda, p_piv, p_rank, &tol, p_work, &info );
	return info;
}

inline integer_t pstrf(
	char uplo, integer_t n, double* A, integer_t lda,
	integer_t* p_piv, integer_t* p_rank, double tol,
	double* p_work)
{
	integer_t info = -1;
	dpstrf( &uplo, &n, A, &lda, p_piv, p_rank, &tol, p_work, &info );
	return info;
}



inline integer_t trcon(
	char norm, char uplo, char diag, integer_t n, const float* A,
	integer_t lda, float* p_rcond, float* p_work, integer_t* p_iwork)
{
	integer_t info = -1;
	strcon(
		&norm, &uplo, &diag, &n, A, &lda, p_rcond, p_work, p_iwork, &info );
	return info;
}

inline integer_t trcon(
	char norm, char uplo, char diag, integer_t n, const double* A,
	integer_t lda, double* p_rcond, double* p_work, integer_t* p_iwork)
{
	integer_t info = -1;
	dtrcon(
		&norm, &uplo, &diag, &n, A, &lda, p_rcond, p_work, p_iwork, &info );
	return info;
}



inline integer_t uncsd2by1(
	char jobu1, char jobu2, char jobv1t,
	integer_t m, integer_t p, integer_t q,
	float* X11, integer_t ldx11,
	float* X21, integer_t ldx21,
	float* theta,
	float* U1, integer_t ldu1,
	float* U2, integer_t ldu2,
	float* V1t, integer_t ldv1t,
	float* work, integer_t lwork, integer_t* iwork)
{
	integer_t info = -1;
	sorcsd2by1(
		&jobu1, &jobu2, &jobv1t,
		&m, &p, &q,
		X11, &ldx11, X21, &ldx21, theta,
		U1, &ldu1, U2, &ldu2, V1t, &ldv1t,
		work, &lwork, iwork, &info);
	return info;
}

inline integer_t uncsd2by1(
	char jobu1, char jobu2, char jobv1t,
	integer_t m, integer_t p, integer_t q,
	double* X11, integer_t ldx11,
	double* X21, integer_t ldx21,
	double* theta,
	double* U1, integer_t ldu1,
	double* U2, integer_t ldu2,
	double* V1t, integer_t ldv1t,
	double* work, integer_t lwork, integer_t* iwork)
{
	integer_t info = -1;
	dorcsd2by1(
		&jobu1, &jobu2, &jobv1t,
		&m, &p, &q,
		X11, &ldx11, X21, &ldx21, theta,
		U1, &ldu1, U2, &ldu2, V1t, &ldv1t,
		work, &lwork, iwork, &info);
	return info;
}



inline integer_t ungqr(
	integer_t m, integer_t n, integer_t k,
	float* A, integer_t lda, const float* p_tau,
	float* p_work, integer_t lwork)
{
	integer_t info = -1;
	sorgqr(&m, &n, &k, A, &lda, p_tau, p_work, &lwork, &info);
	return info;
}

inline integer_t ungqr(
	integer_t m, integer_t n, integer_t k,
	double* A, integer_t lda, const double* p_tau,
	double* p_work, integer_t lwork)
{
	integer_t info = -1;
	dorgqr(&m, &n, &k, A, &lda, p_tau, p_work, &lwork, &info);
	return info;
}



inline integer_t ungrq(
	integer_t m, integer_t n, integer_t k,
	float* A, integer_t lda, float* tau,
	float* work, integer_t lwork)
{
	integer_t info = -1;
	sorgrq(&m, &n, &k, A, &lda, tau, work, &lwork, &info);
	return info;
}

inline integer_t ungrq(
	integer_t m, integer_t n, integer_t k,
	double* A, integer_t lda, double* tau,
	double* work, integer_t lwork)
{
	integer_t info = -1;
	dorgrq(&m, &n, &k, A, &lda, tau, work, &lwork, &info);
	return info;
}

}
