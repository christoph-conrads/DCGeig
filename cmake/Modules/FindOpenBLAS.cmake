# Copyright 2018 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

set(OPENBLAS_ROOT "" CACHE PATH "The OpenBLAS root folder")

if(NOT "${OPENBLAS_ROOT}" STREQUAL "")
	set(_OPENBLAS_PKG_CONFIG_PATH "$ENV{PKG_CONFIG_PATH}")
	set(ENV{PKG_CONFIG_PATH} "${OPENBLAS_ROOT}/lib/pkgconfig:${_OPENBLAS_PKG_CONFIG_PATH}")
endif()

find_package(PkgConfig)
pkg_check_modules(_PC_OPENBLAS REQUIRED openblas)

if(NOT "${OPENBLAS_ROOT}" STREQUAL "")
	set(ENV{PKG_CONFIG_PATH} "${_OPENBLAS_PKG_CONFIG_PATH}")
endif()

find_path(
	OPENBLAS_INCLUDE_DIR
	NAMES openblas_config.h
	PATHS "${_PC_OPENBLAS_INCLUDE_DIRS}"
	NO_DEFAULT_PATH
)
find_library(
	OPENBLAS_LIBRARY
	NAMES openblas
	PATHS "${_PC_OPENBLAS_LIBRARY_DIRS}"
	NO_DEFAULT_PATH
)
set(OPENBLAS_LINKER_FLAGS ${_PC_OPENBLAS_LDFLAGS_OTHER})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
	OpenBLAS
	REQUIRED_VARS OPENBLAS_INCLUDE_DIR OPENBLAS_LIBRARY
)
