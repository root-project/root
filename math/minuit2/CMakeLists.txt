cmake_minimum_required(VERSION 3.1)

option(minuit2_mpi "Enable support for MPI in Minuit2")
option(minuit2_omp "Enable support for OpenMP in Minuit2")

# This package can be built separately
# or as part of ROOT.
if(CMAKE_PROJECT_NAME STREQUAL ROOT)
  ROOT_STANDARD_LIBRARY_PACKAGE(Minuit2
                                HEADERS *.h Minuit2/*.h
                                DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                                DEPENDENCIES MathCore Hist)
endif()

if(minuit2_omp)
  find_package(OpenMP REQUIRED)
  find_package(Threads REQUIRED)

  if(NOT TARGET OpenMP::OpenMP_CXX)
    add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_COMPILE_DEFINITIONS -D_GLIBCXX_PARALLEL)
    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_LIBRARIES})
  endif()

  if(CMAKE_PROJECT_NAME STREQUAL ROOT)
    target_link_libraries(Minuit2 Threads::Threads OpenMP::OpenMP_CXX)
  endif()
endif()

if(minuit2_mpi)
  find_package(MPI REQUIRED)

  if(NOT TARGET MPI::MPI_CXX)
    add_library(MPI::MPI_CXX IMPORTED INTERFACE)
    set_property(TARGET MPI::MPI_CXX PROPERTY INTERFACE_COMPILE_DEFINITIONS ${MPI_CXX_COMPILE_DEFINITIONS})
    set_property(TARGET MPI::MPI_CXX PROPERTY INTERFACE_COMPILE_OPTIONS ${MPI_CXX_COMPILE_FLAGS})
    set_property(TARGET MPI::MPI_CXX PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${MPI_CXX_INCLUDE_DIRS})
    set_property(TARGET MPI::MPI_CXX PROPERTY INTERFACE_LINK_LIBRARIES ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})
  endif()

  if(CMAKE_PROJECT_NAME STREQUAL ROOT)
    target_compile_definitions(Minuit2 PUBLIC MPIPROC)
    target_link_libraries(Minuit2 MPI::MPI_CXX)
  endif()
endif()

if(CMAKE_PROJECT_NAME STREQUAL ROOT)
  add_definitions(-DWARNINGMSG -DUSE_ROOT_ERROR)
  ROOT_ADD_TEST_SUBDIRECTORY(test)
else()
  include(StandAlone.cmake)
endif()
