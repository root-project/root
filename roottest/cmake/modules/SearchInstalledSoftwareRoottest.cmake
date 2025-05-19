include(ExternalProject)

#---Find timeout binary---------------------------------------------------------
if(NOT MSVC)
  find_program(TIMEOUT_BINARY timeout)
endif()

#---Check for MPI---------------------------------------------------------------
if(ROOT_mpi_FOUND)
  message(STATUS "Looking for MPI")
  find_package(MPI)
  if(NOT MPI_FOUND)
    message(FATAL_ERROR "MPI not found. Ensure that the installation of MPI is in the CMAKE_PREFIX_PATH."
      " Example: CMAKE_PREFIX_PATH=<MPI_install_path> (e.g. \"/usr/local/mpich\")")
  endif()
endif()
