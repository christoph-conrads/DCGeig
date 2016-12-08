# Copyright 2016 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

IF(USE_MKL)
	include_directories(${MKL_INCLUDE_DIR})
	set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${MKL_LIBRARY_PATH})

	find_library(MKL_RT "mkl_rt")
	find_library(PTHREAD "pthread")
	find_library(LIBM "m")

	set(LINALG_LIBS ${MKL_RT} ${PTHREAD} ${LIBM})
	set(CMAKE_SHARED_LINKER_FLAGS
		" ${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-as-needed")
ELSE(USE_MKL)
	find_package(BLAS REQUIRED)
	find_package(LAPACK REQUIRED)
	set(LINALG_LIBS ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
ENDIF(USE_MKL)


add_library(solvers SHARED
	deflate_gep.cpp
	ggsvdcs.cpp
	solve_gep_with_deflation.cpp
	solve_gep_with_gsvd.cpp
	solve_gep_with_qr_csd.cpp)
target_include_directories(solvers PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(solvers ${LINALG_LIBS})
set_target_properties(solvers PROPERTIES OUTPUT_NAME "hpsd_gep_solvers")

install(TARGETS solvers DESTINATION lib)
