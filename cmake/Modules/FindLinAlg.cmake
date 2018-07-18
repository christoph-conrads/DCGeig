# Copyright 2018 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

set(BLAS "openblas" CACHE STRING "Select the BLAS implementation")

if("${BLAS}" STREQUAL "openblas")
	find_package(OpenBLAS REQUIRED)
	set(LINALG_LIBRARIES ${OPENBLAS_LIBRARY})
	set(LINALG_INCLUDE_DIRS ${OPENBLAS_INCLUDE_DIR})
	set(LINALG_LINKER_FLAGS ${OPENBLAS_LINKER_FLAGS})
elseif("${BLAS}" STREQUAL "mkl")
	find_package(MKL REQUIRED)
	set(LINALG_LIBRARIES ${MKL_LIBRARIES})
	set(LINALG_INCLUDE_DIRS ${MKL_INCLUDE_DIRS})
	set(LINALG_LINKER_FLAGS ${MKL_LINKER_FLAGS})
else()
	message(FATAL_ERROR "Unknown BLAS implementation '${BLAS}'")
endif()
