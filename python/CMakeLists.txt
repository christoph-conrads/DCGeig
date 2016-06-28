find_package(PythonInterp 2 REQUIRED)
find_package(PythonLibs 2 REQUIRED)
find_program(CYTHON_EXECUTABLE cython)
find_library(METIS_LIBRARIES metis)


if(NOT METIS_LIBRARIES)
	message(FATAL_ERROR "metis not found")
endif(NOT METIS_LIBRARIES)


set(PYTHON_DIR "python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
set(PACKAGE_DIR "lib/${PYTHON_DIR}/site-packages/${PROJECT_NAME}")

add_subdirectory(dcgeig)


execute_process(
	COMMAND ${GIT_EXECUTABLE} rev-parse --quiet HEAD
	OUTPUT_VARIABLE GIT_COMMIT
	WORKING_DIRECTORY CMAKE_SOURCE_DIR)


# Python stuff
set(CYTHON_CFLAGS -w -fwrapv -fno-strict-aliasing)

execute_process(
	COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; import sys; sys.stdout.write(numpy.get_include())"
	OUTPUT_VARIABLE NUMPY_INCLUDE_DIRS)


# build modules
add_custom_command(
	OUTPUT metis.c
	COMMAND ${CYTHON_EXECUTABLE}
		--output-file
		${CMAKE_CURRENT_BINARY_DIR}/metis.c
		${CMAKE_CURRENT_SOURCE_DIR}/metis.pyx
	MAIN_DEPENDENCY metis.pyx)

add_library(metis SHARED metis.c)
set_target_properties(metis PROPERTIES PREFIX "")
target_compile_options(metis PUBLIC ${CYTHON_CFLAGS})
target_include_directories(metis PUBLIC ${PYTHON_INCLUDE_DIRS})
target_include_directories(metis PUBLIC ${NUMPY_INCLUDE_DIRS})
target_include_directories(metis PUBLIC ${NUMPY_INCLUDE_DIRS})
target_link_libraries(metis PUBLIC ${PYTHON_LIBRARIES})
target_link_libraries(metis PUBLIC ${METIS_LIBRARIES})


add_custom_command(
	OUTPUT solvers.c
	COMMAND ${CYTHON_EXECUTABLE}
		--output-file ${CMAKE_CURRENT_BINARY_DIR}/solvers.c
		${CMAKE_CURRENT_SOURCE_DIR}/solvers.pyx
	MAIN_DEPENDENCY solvers.pyx)

add_library(pysolvers SHARED solvers.c)
set_target_properties(pysolvers PROPERTIES OUTPUT_NAME "solvers" PREFIX "")
target_compile_options(pysolvers PUBLIC ${CYTHON_CFLAGS})
target_include_directories(pysolvers PUBLIC ${PYTHON_INCLUDE_DIRS})
target_include_directories(pysolvers PUBLIC ${NUMPY_INCLUDE_DIRS})
target_link_libraries(pysolvers PUBLIC ${PYTHON_LIBRARIES})
target_include_directories(metis PUBLIC ${NUMPY_INCLUDE_DIRS})
target_link_libraries(pysolvers PUBLIC ${METIS_LIBRARIES})
target_link_libraries(pysolvers PUBLIC solvers)


install(TARGETS metis pysolvers LIBRARY DESTINATION ${PACKAGE_DIR})
install(PROGRAMS multilevel-gep-solver solve-gep DESTINATION bin)
