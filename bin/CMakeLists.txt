find_package(Git)


if(GIT_FOUND)
	message("git found: ${GIT_EXECUTABLE}")
endif(GIT_FOUND)


execute_process(
	COMMAND ${GIT_EXECUTABLE} rev-parse --quiet HEAD
	OUTPUT_VARIABLE GIT_COMMIT
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})


configure_file("benchmark-gep-solver.in" "benchmark-gep-solver" @ONLY)

install(
	PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/benchmark-gep-solver
	DESTINATION bin)
