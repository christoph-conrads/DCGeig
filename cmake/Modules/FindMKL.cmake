# Copyright 2018 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

set(MKL_ROOT "/opt/intel/mkl" CACHE PATH "The Intel MKL root folder")

find_path(
	MKL_INCLUDE_DIR
	NAMES mkl.h
	HINTS "${MKL_ROOT}"
	PATH_SUFFIXES include
)
find_library(
	MKL_LIBRARY
	NAMES mkl_rt
	HINTS "${MKL_ROOT}"
	PATH_SUFFIXES "lib/intel64"
)

find_library(LIBPTHREAD NAMES pthread)
find_library(LIBM NAMES m)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
	MKL
	REQUIRED_VARS MKL_INCLUDE_DIR MKL_LIBRARY LIBPTHREAD LIBM
)

if(MKL_FOUND)
	set(MKL_LIBRARIES ${MKL_LIBRARY} ${LIBPTHREAD} ${LIBPTHREAD})
	set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
	set(MKL_LINKER_FLAGS "-Wl,--no-as-needed")
endif()

mark_as_advanced(MKL_INCLUDE_DIR MKL_LIBRARY MKL_LINKER_FLAGS)
