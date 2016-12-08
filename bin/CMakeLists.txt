# Copyright 2016 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

find_package(Git)


execute_process(
	COMMAND ${GIT_EXECUTABLE} rev-parse --quiet HEAD
	OUTPUT_VARIABLE GIT_COMMIT
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})


configure_file("benchmark-gep-solver.in" "benchmark-gep-solver" @ONLY)

install(
	PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/benchmark-gep-solver
	DESTINATION bin)
