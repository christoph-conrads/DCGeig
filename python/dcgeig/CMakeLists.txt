add_subdirectory(tests)


configure_file("__init__.py.in" "__init__.py")

install(FILES
		${CMAKE_CURRENT_BINARY_DIR}/__init__.py
		binary_tree.py
		error_analysis.py
		gallery.py
		options.py
		polynomial.py
		solver.py
		sparse_tools.py
		subspace_iteration.py
		utils.py
	DESTINATION ${PACKAGE_DIR})
