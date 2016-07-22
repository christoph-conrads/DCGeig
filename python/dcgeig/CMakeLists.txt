add_subdirectory(tests)


configure_file("__init__.py.in" "__init__.py")

install(FILES
		${CMAKE_CURRENT_BINARY_DIR}/__init__.py
		direct_substructuring.py
		error_analysis.py
		multilevel_solver.py
		multilevel_tools.py
		sparse_tools.py
		subspace_iteration.py
		stats.py
		utils.py
	DESTINATION ${PACKAGE_DIR})
