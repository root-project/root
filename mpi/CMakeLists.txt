SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#################
#MPI ENVIRONMENT#
#################
FIND_PACKAGE(MPI REQUIRED)

#Standalone Build
if(NOT CMAKE_PROJECT_NAME STREQUAL ROOT)
  cmake_minimum_required(VERSION 3.4.3 FATAL_ERROR)
  project(ROOTMpi)
  set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../cmake/modules)
  include(RootBuildOptions)
  include(RootNewMacros)
  include(CheckCompiler)
  include(MacroEnsureVersion)
  include(StandaloneBuild)
  #---Check for Python installation-------------------------------------------------------
  if(python OR python2 OR python3)
  message(STATUS "Looking for Python")
  #---First look for the python interpreter and fix the version of it for the libraries--
  if(python3)
    find_package(PythonInterp 3.5)
  else()
    find_package(PythonInterp)
  endif()
  if(PYTHONINTERP_FOUND)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys;sys.stdout.write(sys.version[:3])"
                    OUTPUT_VARIABLE PYTHON_VERSION)
    message(STATUS "Found Python interpreter version ${PYTHON_VERSION}")
  endif()
  set(Python_ADDITIONAL_VERSIONS ${PYTHON_VERSION})
  endif()

  if(testing)  
    configure_file(${CMAKE_SOURCE_DIR}/../cmake/modules/CTestCustom.cmake ${CMAKE_BINARY_DIR} COPYONLY)
    enable_testing()
    include(RootCTest)
    add_subdirectory(test)
  endif()
  
  string(REPLACE "/bin/mpiexec" "" MPI_PATH "${MPIEXEC}")
  file( WRITE  "${CMAKE_BINARY_DIR}/env.sh" "export PATH=${MPI_PATH}/bin:$PATH\n")
  file( APPEND "${CMAKE_BINARY_DIR}/env.sh" "export PATH=$PATH:${CMAKE_BINARY_DIR}/rootmpi\n")
  
  file( APPEND "${CMAKE_BINARY_DIR}/env.sh"  "if [ -n \"$ROOT_LIBRARY_PATH\" ] ; then\n")
  file( APPEND "${CMAKE_BINARY_DIR}/env.sh" "\texport ROOT_LIBRARY_PATH=$ROOT_LIBRARY_PATH:${CMAKE_BINARY_DIR}/rmpi\n")
  file( APPEND "${CMAKE_BINARY_DIR}/env.sh"  "else\n")
  file( APPEND "${CMAKE_BINARY_DIR}/env.sh" "\texport ROOT_LIBRARY_PATH=${CMAKE_BINARY_DIR}/rmpi\n")
  file( APPEND "${CMAKE_BINARY_DIR}/env.sh"  "fi\n")

  file( APPEND "${CMAKE_BINARY_DIR}/env.sh"  "if [ -n \"$LD_LIBRARY_PATH\" ] ; then\n")
  file( APPEND "${CMAKE_BINARY_DIR}/env.sh" "\texport LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CMAKE_BINARY_DIR}/rmpi:${MPI_PATH}/lib\n")
  file( APPEND "${CMAKE_BINARY_DIR}/env.sh"  "else\n")
  file( APPEND "${CMAKE_BINARY_DIR}/env.sh" "\texport LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/rmpi:${MPI_PATH}/lib\n")
  file( APPEND "${CMAKE_BINARY_DIR}/env.sh"  "fi\n")
  
  file( APPEND "${CMAKE_BINARY_DIR}/env.sh"  "if [ -n \"$ROOT_INCLUDE_PATH\" ] ; then\n")
  file( APPEND "${CMAKE_BINARY_DIR}/env.sh" "\texport ROOT_INCLUDE_PATH=$ROOT_INCLUDE_PATH:${CMAKE_SOURCE_DIR}/rmpi/inc:${MPI_PATH}/include\n")
  file( APPEND "${CMAKE_BINARY_DIR}/env.sh"  "else\n")
  file( APPEND "${CMAKE_BINARY_DIR}/env.sh" "\texport ROOT_INCLUDE_PATH=${CMAKE_SOURCE_DIR}/rmpi/inc:${MPI_PATH}/include\n")
  file( APPEND "${CMAKE_BINARY_DIR}/env.sh"  "fi\n")

  get_property(__allHeaders GLOBAL PROPERTY ROOT_HEADER_TARGETS)
  add_custom_target(move_headers ALL DEPENDS ${__allHeaders})  
endif()



include_directories(${MPI_CXX_INCLUDE_PATH} ${MPI_C_INCLUDE_PATH} "${CMAKE_BINARY_DIR}/include")
ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS  -Wno-overloaded-virtual  -Wno-literal-suffix)

add_subdirectory(rmpi)
add_subdirectory(rootmpi)

if(testing)  
ROOT_ADD_TEST_SUBDIRECTORY(test)
endif()
