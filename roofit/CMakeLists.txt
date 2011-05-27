if(NOT CMAKE_PROJECT_NAME STREQUAL ROOT)
  cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
  project(RooFit)
  include(${ROOTSYS}/cmake/modules/StandaloneBuild.cmake)
endif()

ROOT_USE_PACKAGE(math/mathcore)
ROOT_USE_PACKAGE(math/foam)
ROOT_USE_PACKAGE(math/minuit)
ROOT_USE_PACKAGE(proof/proof)

if(mathmore)
  ROOT_USE_PACKAGE(math/mathmore)
  set(ROOT_MATHMORE_LIBRARY MathMore)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/roofit/inc) 
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/roofitcore/inc) 
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/roostats/inc) 

add_subdirectory(roofitcore)
add_subdirectory(roofit)
add_subdirectory(roostats)
add_subdirectory(histfactory)


