if(NOT CMAKE_PROJECT_NAME STREQUAL ROOT)
  cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
  project(RooFit)
  find_package(ROOT REQUIRED)
  include(${ROOT_USE_FILE})
endif()

ROOT_USE_PACKAGE(math/mathcore)
ROOT_USE_PACKAGE(math/foam)
ROOT_USE_PACKAGE(math/minuit)
ROOT_USE_PACKAGE(math/smatrix)
ROOT_USE_PACKAGE(proof/proof)

if(mathmore OR ROOT_mathmore_FOUND)
  set(ROOT_MATHMORE_LIBRARY MathMore)
  ROOT_USE_PACKAGE(math/mathmore)	
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/roofit/inc) 
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/roofitcore/inc) 
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/roostats/inc) 

add_subdirectory(roofitcore)
add_subdirectory(roofit)
add_subdirectory(roostats)
if(xml OR ROOT_xml_FOUND)
  add_subdirectory(histfactory)
endif()


