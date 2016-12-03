# Copyright 2016 Christoph Conrads
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

install(FILES
		__init__.py
		test_binary_tree.py
		test_error_analysis.py
		test_gallery.py
		test_linalg.py
		test_metis.py
		test_polynomial.py
		test_solver.py
		test_solvers.py
		test_sparse_tools.py
		test_subspace_iteration.py
		test_utils.py
	DESTINATION ${PACKAGE_DIR}/tests)
