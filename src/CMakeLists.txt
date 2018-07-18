# Copyright 2016, 2018 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

find_package(LinAlg REQUIRED)

add_library(solvers SHARED
	deflate_gep.cpp
	ggsvdcs.cpp
	solve_gep_with_deflation.cpp
	solve_gep_with_gsvd.cpp
	solve_gep_with_qr_csd.cpp)
target_include_directories(solvers PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_include_directories(solvers PUBLIC ${LINALG_INCLUDE_DIRS})
target_link_libraries(solvers PUBLIC ${LINALG_LIBRARIES})
target_link_libraries(solvers INTERFACE ${LINALG_LINKER_FLAGS})
set_target_properties(solvers PROPERTIES OUTPUT_NAME "hpsd_gep_solvers")

install(TARGETS solvers DESTINATION lib)
