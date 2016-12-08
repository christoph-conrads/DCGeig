/*
 * Copyright 2015-2016 Christoph Conrads
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <stdint.h>


extern "C"
{

void sgemm_(
	const char* transa, const char* transb,
	const int32_t* m, const int32_t* n, const int32_t* k, const float* alpha,
	const float* A, const int32_t* lda, const float* B, const int32_t* ldb,
	const float* beta, float* C, const int32_t* ldc);

void dgemm_(
	const char* transa, const char* transb,
	const int32_t* m, const int32_t* n, const int32_t* k, const double* alpha,
	const double* A, const int32_t* lda, const double* B, const int32_t* ldb,
	const double* beta, double* C, const int32_t* ldc);



void ssymm_(
	const char* side, const char* uplo,
	const int32_t* m, const int32_t* n, const float* alpha,
	const float* A, const int32_t* lda, const float* B, const int32_t* ldb,
	const float* beta, float* C, const int32_t* ldc);

void dsymm_(
	const char* side, const char* uplo,
	const int32_t* m, const int32_t* n, const double* alpha,
	const double* A, const int32_t* lda, const double* B, const int32_t* ldb,
	const double* beta, double* C, const int32_t* ldc);



void strsm_(
	const char* side, const char* uplo, const char* transa, const char* diag,
	const int32_t* m, const int32_t* n, const float* alpha,
	const float* A, const int32_t* lda, float* B, const int32_t* ldb);

void dtrsm_(
	const char* side, const char* uplo, const char* transa, const char* diag,
	const int32_t* m, const int32_t* n, const double* alpha,
	const double* A, const int32_t* lda, double* B, const int32_t* ldb);

} // extern "C"



namespace blas
{
	typedef int32_t integer_t;



inline void gemm(
	char transa, char transb, integer_t m, integer_t n, integer_t k,
	float alpha, const float* A, integer_t lda, const float* B, integer_t ldb,
	float beta, float* C, integer_t ldc)
{
	sgemm_(
		&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

inline void gemm(
	char transa, char transb, integer_t m, integer_t n, integer_t k,
	double alpha, const double* A, integer_t lda, const double* B,integer_t ldb,
	double beta, double* C, integer_t ldc)
{
	dgemm_(
		&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}



inline void hemm(
	char side, char uplo, integer_t m, integer_t n, float alpha,
	const float* A, integer_t lda, const float* B, integer_t ldb,
	float beta, float* C, integer_t ldc)
{
	ssymm_(&side, &uplo, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

inline void hemm(
	char side, char uplo, integer_t m, integer_t n, double alpha,
	const double* A, integer_t lda, const double* B, integer_t ldb,
	double beta, double* C, integer_t ldc)
{
	dsymm_(&side, &uplo, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}



inline void trsm(
	char side, char uplo, char transa, char diag,
	integer_t m, integer_t n, float alpha,
	const float* A, integer_t lda, float* B, integer_t ldb)
{
	strsm_( &side, &uplo, &transa, &diag, &m, &n, &alpha, A, &lda, B, &ldb );
}

inline void trsm(
	char side, char uplo, char transa, char diag,
	integer_t m, integer_t n, double alpha,
	const double* A, integer_t lda, double* B, integer_t ldb)
{
	dtrsm_( &side, &uplo, &transa, &diag, &m, &n, &alpha, A, &lda, B, &ldb );
}


}
