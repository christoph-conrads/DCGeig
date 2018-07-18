# Copyright 2016, 2018 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

set(METIS_ROOT "" CACHE PATH "The METIS root folder")

find_path(
	METIS_INCLUDE_DIR
	NAMES metis.h
	HINTS "${METIS_ROOT}"
	PATH_SUFFIXES include
)
find_library(
	METIS_LIBRARY
	NAMES metis
	HINTS "${METIS_ROOT}"
	PATH_SUFFIXES lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
	Metis
	REQUIRED_VARS METIS_LIBRARY METIS_INCLUDE_DIR)

if(METIS_FOUND)
	set(METIS_LIBRARIES ${METIS_LIBRARY})
	set(METIS_INCLUDE_DIRS ${METIS_INCLUDE_DIR})
endif()

mark_as_advanced(METIS_INCLUDE_DIR METIS_LIBRARY)
