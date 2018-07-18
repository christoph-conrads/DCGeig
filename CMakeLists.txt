# Copyright 2016, 2018 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.2.0)
project(dcgeig VERSION 2.0.0)


# set variables
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wall -std=c++11 -pedantic")


# command line options
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
