
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include_directories(inc/)
add_definitions(-w "-DROOT_MPI_EXEC=\"${MPIEXEC}\"")

SET(ROOTMPI_SOURCES src/main.cxx src/TRootMpi.cxx)
add_executable(rootmpi ${ROOTMPI_SOURCES})
target_link_libraries(rootmpi Core Rint)
install(TARGETS rootmpi DESTINATION bin)
