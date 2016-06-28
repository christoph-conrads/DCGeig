cmake_minimum_required(VERSION 3.2.0)
project(dcgeig VERSION 1.2.0)


# set variables
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wall -std=c++11 -pedantic")


# options
option(USE_MKL "Link against Intel MKL" OFF)
set(MKL_INCLUDE_DIR "/opt/intel/mkl/include"
	CACHE PATH "Path to Intel MKL include directory")
set(MKL_LIBRARY_PATH "/opt/intel/mkl/lib/intel64"
	CACHE PATH "Path to Intel MKL libraries")
option(BE_QUIET "Tell CMake to be quiet" OFF)
option(BUILD_TESTING "Build tests" OFF)


IF(BE_QUIET)
	set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)
ENDIF(BE_QUIET)


IF(BUILD_TESTING)
	enable_testing()
ENDIF(BUILD_TESTING)


# add subdirectories
include_directories("${CMAKE_BINARY_DIR}/include")

add_subdirectory(bin)
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(python)
add_subdirectory(tests)
