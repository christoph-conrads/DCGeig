if(USE_MKL)
	set(HAS_GGSVD3 False)
else(USE_MKL)
	set(HAS_GGSVD3 True)
endif(USE_MKL)

configure_file(config.h.in config.h)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/config.h DESTINATION include)
install(FILES hpsd_gep_solvers.h DESTINATION include)
install(FILES hpsd_gep_solvers.hpp DESTINATION include)
