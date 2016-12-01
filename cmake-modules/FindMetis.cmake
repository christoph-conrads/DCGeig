find_path(METIS_INCLUDE_DIR NAMES metis.h)
find_library(METIS_LIBRARY NAMES metis)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
	Metis
	REQUIRED_VARS METIS_LIBRARY METIS_INCLUDE_DIR)

if(METIS_FOUND)
	set(METIS_LIBRARIES ${METIS_LIBRARY})
	set(METIS_INCLUDE_DIRS ${METIS_INCLUDE_DIR})
endif()

mark_as_advanced(METIS_INCLUDE_DIR METIS_LIBRARY)
