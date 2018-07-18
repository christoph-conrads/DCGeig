# Copyright 2016, 2018 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

if("${BLAS}" STREQUAL "mkl")
	set(HAS_INTEL_MKL True)
	set(HAS_GGSVD3 False)
else()
	set(HAS_INTEL_MKL False)
	set(HAS_GGSVD3 True)
endif()

configure_file(config.h.in config.h)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/config.h DESTINATION include)
install(FILES hpsd_gep_solvers.h DESTINATION include)
install(FILES hpsd_gep_solvers.hpp DESTINATION include)
