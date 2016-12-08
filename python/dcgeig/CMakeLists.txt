# Copyright 2016 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

add_subdirectory(tests)


configure_file("__init__.py.in" "__init__.py")

install(FILES
		${CMAKE_CURRENT_BINARY_DIR}/__init__.py
		binary_tree.py
		error_analysis.py
		gallery.py
		linalg.py
		options.py
		polynomial.py
		solver.py
		sparse_tools.py
		subspace_iteration.py
		utils.py
	DESTINATION ${PACKAGE_DIR})
