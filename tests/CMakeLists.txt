if(NOT BUILD_TESTING)
	return()
endif()


# find packages, libs
find_package(Boost 1.36.0 REQUIRED COMPONENTS unit_test_framework)


# target
add_definitions(-DBOOST_TEST_DYN_LINK)

file(GLOB SOURCES "*.cpp")
include_directories(${PROJECT_SOURCE_DIR} ${Boost_INCLUDE_DIR})

add_executable(cpp_test ${SOURCES})

target_link_libraries(cpp_test solvers ${Boost_LIBRARIES})


# run tests
add_test(NAME valgrind_cpp_test COMMAND valgrind --quiet ./cpp_test)
