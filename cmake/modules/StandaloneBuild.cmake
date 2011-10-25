##################################################################################################
# --- StandaloneBuild.cmake module----------------------------------------------------------------
# This module provides the sufficent environemnt to be able to build ROOT as standalone projects
# It only assumes a valid ROOTSYS that can be provided directly with the command line
#              cmake -DROOTSYS=<ROOT Installation> <package source> 
##################################################################################################
 
#---Find ROOT ------------------------------------------------------------------------------------
if(DEFINED ROOTSYS AND NOT DEFINED ROOT_DIR)
  set(ROOT_DIR ${ROOTSYS}/cmake)
endif()
find_package(ROOT REQUIRED)

#---Minimal environment---------------------------------------------------------------------------
include(${ROOT_USE_FILE})

#---Initialize project----------------------------------------------------------------------------
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#---Set Link and include directories--------------------------------------------------------------
set(CMAKE_INCLUDE_DIRECTORIES_BEFORE ON)
include_directories(${ROOT_INCLUDE_DIRS})
link_directories(${ROOT_LIBRARY_DIRS})



