/*
 * Copyright 2015-2016 Christoph Conrads
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 * This Source Code Form is "Incompatible With Secondary Licenses", as
 * defined by the Mozilla Public License, v. 2.0.
 */
#include <boost/test/unit_test.hpp>

#include <hpsd_gep_solvers.hpp>
#include <lapack.hpp>
#include <matrix.hpp>

#include <limits>
#include <memory>
#include <cmath>


using namespace std;
using lapack::integer_t;

typedef std::vector<double> DoubleList;
typedef std::vector<lapack::integer_t> IntegerList;



struct Fixture
{
	Fixture(std::size_t n, std::size_t lwork, std::size_t liwork) :
		nan(std::numeric_limits<double>::quiet_NaN()),
		K(n*n, nan),
		M(n*n, nan),
		X(n*n, nan),
		lambda(n, nan),
		work(lwork, nan),
		iwork(liwork, -1)
	{
	}


	const double nan;
	DoubleList K;
	DoubleList M;
	DoubleList X;
	DoubleList lambda;
	DoubleList work;
	IntegerList iwork;
};



BOOST_AUTO_TEST_CASE(qr_csd_test_access_upper_triangle)
{
	const size_t n = 8;

	const size_t lwork_min = 18;
	const size_t lwork = 3*n*n + max(lwork_min, 17*n - 4u);

	const size_t liwork = 3*n;

	Fixture fixture(n, lwork, liwork);

	double* const K = fixture.K.data();
	double* const M = fixture.M.data();
	double* const lambda = fixture.lambda.data();

	integer_t rank = -1;

	lapack::laset('U', n, n, 0.0, 1.0, K, n);
	lapack::laset('U', n, n, 0.0, 1.0, M, n);

	integer_t ret = solve_gep_with_qr_csd(
		n, K, n, M, n, lambda, &rank,
		fixture.work.data(), lwork, fixture.iwork.data(), liwork
	);

	BOOST_REQUIRE_EQUAL( ret, 0 );

	BOOST_CHECK_EQUAL( rank, n );

	for(size_t i = 0; i < n; ++i)
		BOOST_CHECK( !std::isnan(lambda[i]) );

	for(size_t i = 0; i < n*n; ++i)
		BOOST_CHECK( !std::isnan(K[i]) );
}



BOOST_AUTO_TEST_CASE(gsvd_test_access_upper_triangle)
{
	const size_t n = 8;

	const size_t lwork = n*n + 6*n + 1;
	const size_t liwork = n;

	Fixture fixture(n, lwork, liwork);

	double* const K = fixture.K.data();
	double* const M = fixture.M.data();
	double* const lambda = fixture.lambda.data();

	lapack::laset('U', n, n, 0.0, 1.0, K, n);
	lapack::laset('U', n, n, 0.0, 1.0, M, n);

	integer_t rank = -1;

	integer_t ret = solve_gep_with_gsvd(
		n, K, n, M, n, lambda, &rank,
		fixture.work.data(), lwork, fixture.iwork.data(), n
	);

	BOOST_REQUIRE_EQUAL( ret, 0 );

	BOOST_CHECK_EQUAL( rank, n );

	for(size_t i = 0; i < n; ++i)
		BOOST_CHECK( !std::isnan(lambda[i]) );

	for(size_t i = 0; i < n*n; ++i)
		BOOST_CHECK( !std::isnan(K[i]) );
}



BOOST_AUTO_TEST_CASE(test_deflate_gep)
{
	const size_t n = 8;
	const double NaN = std::numeric_limits<double>::quiet_NaN();

	const size_t minimum_lwork = 2*n*n + 6*n + 1;
	const size_t minimum_liwork = 5*n + 3;

	Fixture fixture(n, minimum_lwork, minimum_liwork);
	DoubleList Qlist( n*n, NaN );

	double* const K = fixture.K.data(); const integer_t ldk = n;
	double* const M = fixture.M.data(); const integer_t ldm = n;
	double* const lambda = fixture.lambda.data();
	integer_t rank_M = -1;
	double* const X = fixture.X.data(); const integer_t ldx = n;
	double* const Q = Qlist.data(); const integer_t ldq = n;

	double* const work = fixture.work.data();
	integer_t* const iwork = fixture.iwork.data();

	lapack::laset('U', n, n, 0.0, 1.0, K, n);
	lapack::laset('U', n, n, 0.0, 1.0, M, n);

	const integer_t ret = deflate_gep(
		n, K, ldk, M, ldm, lambda, &rank_M, X, ldx, Q, ldq,
		work, minimum_lwork, iwork, minimum_liwork);

	BOOST_REQUIRE_EQUAL( ret, 0 );
	BOOST_CHECK_EQUAL( rank_M, n );

	for(size_t i = 0; i < n*n; ++i)
		BOOST_CHECK( !std::isnan(X[i]) );
}



BOOST_AUTO_TEST_CASE(test_deflate_gep_zero_M)
{
	const size_t n = 8;
	const double NaN = std::numeric_limits<double>::quiet_NaN();

	const size_t minimum_lwork = 2*n*n + 6*n + 1;
	const size_t minimum_liwork = 5*n + 3;

	Fixture fixture(n, minimum_lwork, minimum_liwork);
	DoubleList Qlist( n*n, NaN );

	double* const K = fixture.K.data(); const integer_t ldk = n;
	double* const M = fixture.M.data(); const integer_t ldm = n;
	double* const lambda = fixture.lambda.data();
	integer_t rank_M = -1;
	double* const X = fixture.X.data(); const integer_t ldx = n;
	double* const Q = Qlist.data(); const integer_t ldq = n;

	double* const work = fixture.work.data();
	integer_t* const iwork = fixture.iwork.data();

	lapack::laset('U', n, n, 0.0, 1.0, K, n);
	lapack::laset('U', n, n, 0.0, 0.0, M, n);

	const integer_t ret = deflate_gep(
		n, K, ldk, M, ldm, lambda, &rank_M, X, ldx, Q, ldq,
		work, minimum_lwork, iwork, minimum_liwork);

	BOOST_REQUIRE_EQUAL( ret, 0 );
	BOOST_CHECK_EQUAL( rank_M, 0 );

	for(size_t i = 1; i <= n; ++i)
		for(size_t j = i; j <= n; ++j)
		{
			const std::size_t k = compute_offset(n, ldm, i, j);
			BOOST_CHECK_EQUAL( M[k], 0 );
		}

	for(size_t i = 0; i < n*n; ++i)
		BOOST_CHECK( !std::isnan(X[i]) );
}



BOOST_AUTO_TEST_CASE(test_deflate_gep_singular_M)
{
	const size_t n = 8;
	const double NaN = std::numeric_limits<double>::quiet_NaN();

	const size_t minimum_lwork = 2*n*n + 6*n + 1;
	const size_t minimum_liwork = 5*n + 3;

	Fixture fixture(n, minimum_lwork, minimum_liwork);
	DoubleList Qlist( n*n, NaN );

	double* const K = fixture.K.data(); const integer_t ldk = n;
	double* const M = fixture.M.data(); const integer_t ldm = n;
	double* const lambda = fixture.lambda.data();
	integer_t rank_M = -1;
	double* const X = fixture.X.data(); const integer_t ldx = n;
	double* const Q = Qlist.data(); const integer_t ldq = n;

	double* const work = fixture.work.data();
	integer_t* const iwork = fixture.iwork.data();

	lapack::laset('U', n, n, 0.0, 1.0, K, n);
	lapack::laset('U', n, n, 0.0, 1.0, M, n);
	M[0] = 0;

	const integer_t ret = deflate_gep(
		n, K, ldk, M, ldm, lambda, &rank_M, X, ldx, Q, ldq,
		work, minimum_lwork, iwork, minimum_liwork);

	BOOST_REQUIRE_EQUAL( ret, 0 );
	BOOST_CHECK_EQUAL( rank_M, n-1 );

	for(size_t i = 0; i < (n-1)*(n-1); ++i)
		BOOST_CHECK( !std::isnan(K[i]) );

	for(size_t i = 0; i < (n-1)*(n-1); ++i)
		BOOST_CHECK( !std::isnan(M[i]) );

	for(size_t i = 0; i < n; ++i)
		BOOST_CHECK( !std::isnan(lambda[i]) );

	for(size_t i = 0; i < n*n; ++i)
		BOOST_CHECK( !std::isnan(X[i]) );

	for(size_t i = 0; i < n*n; ++i)
		BOOST_CHECK( !std::isnan(Q[i]) );
}



BOOST_AUTO_TEST_CASE(deflation_test_access_upper_triangle)
{
	const size_t n = 8;
	const size_t lwork = 4*n*n + 6*n + 1;
	const size_t liwork = 5*n + 3;

	Fixture fixture(n, lwork, liwork);

	double* const K = fixture.K.data();
	double* const M = fixture.M.data();
	double* const lambda = fixture.lambda.data();

	lapack::laset('U', n, n, 0.0, 1.0, K, n);
	lapack::laset('U', n, n, 0.0, 1.0, M, n);

	integer_t rank_M = -1;
	double* const work = fixture.work.data();
	integer_t* const iwork = fixture.iwork.data();

	integer_t ret = solve_gep_with_deflation(
		n, K, n, M, n, lambda, &rank_M, work, lwork, iwork, liwork);

	BOOST_REQUIRE_EQUAL( ret, 0 );

	for(size_t i = 0; i < n; ++i)
		BOOST_CHECK( !std::isnan(lambda[i]) );

	for(size_t i = 0; i < n*n; ++i)
		BOOST_REQUIRE( !std::isnan(K[i]) );
}
