
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include_directories(inc/)

find_program(VALGRIND "valgrind")
if(VALGRIND)
    message(STATUS "Found ${VALGRIND}")
    add_definitions("-DROOT_MPI_VALGRIND=\"${VALGRIND}\"" )
    add_definitions("-DROOT_MPI_VALGRINDFOUND" )
endif()

if(python)
    if(PYTHONINTERP_FOUND)
        add_definitions("-DPYTHONINTERP_FOUND" )
        add_definitions("-DPYTHON_EXECUTABLE=\"${PYTHON_EXECUTABLE}\"" )
    endif()
endif()

add_definitions("-DROOT_MPI_EXEC=\"${MPIEXEC}\"" "-DROOT_MPI_CXX=\"${CXX}\"" "-DROOT_MPI_ROOTSYS=\"$ENV{ROOTSYS}\"")

if(${MPI_CXX_FOUND})
add_definitions( "-DROOT_MPI_CFLAGS=\"${MPI_CXX_COMPILE_FLAGS}\""  "-DROOT_MPI_LDFLAGS=\"${MPI_CXX_LINK_FLAGS}\"")
else()
add_definitions( "-DROOT_MPI_CFLAGS=\"${MPI_C_COMPILE_FLAGS}\""  "-DROOT_MPI_LDFLAGS=\"${MPI_C_LINK_FLAGS}\"")
endif()

SET(ROOTMPI_SOURCES src/main.cxx src/TRootMpi.cxx)
add_executable(rootmpi ${ROOTMPI_SOURCES})
target_link_libraries(rootmpi Core Rint)
install(TARGETS rootmpi DESTINATION bin)

