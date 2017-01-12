SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(NOT CMAKE_PROJECT_NAME STREQUAL ROOT)
  cmake_minimum_required(VERSION 3.4.3 FATAL_ERROR)
  project(ROOTMpi)
  set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../cmake/modules)
  include(RootBuildOptions)
  include(RootNewMacros)
  include(CheckCompiler)
  include(MacroEnsureVersion)
  include(StandaloneBuild)
  if(testing)  
    configure_file(${CMAKE_SOURCE_DIR}/../cmake/modules/CTestCustom.cmake ${CMAKE_BINARY_DIR} COPYONLY)
    enable_testing()
    include(RootCTest)
    add_subdirectory(test)
  endif()
endif()



#################
#MPI ENVIRONMENT#
#################
FIND_PACKAGE(MPI REQUIRED)

include_directories(${MPI_CXX_INCLUDE_PATH})
ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-overloaded-virtual ${MPI_CXX_COMPILE_FLAGS} -Wno-literal-suffix)

add_subdirectory(rmpi)
add_subdirectory(rootmpi)

if(testing)  
ROOT_ADD_TEST_SUBDIRECTORY(test)
endif()
