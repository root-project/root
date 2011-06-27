##################################################################################################
# --- StandaloneBuild.cmake module----------------------------------------------------------------
# This module provides the sufficent environemnt to be able to build ROOT as standalone projects
# It only assumes a valid ROOTSYS that can be provided directly with the command line
#              cmake -DROOTSYS=<ROOT Installation> <package source> 
##################################################################################################
 
#---Set the CMake module path to locate the needed modules ---------------------------------------
set(CMAKE_MODULE_PATH ${ROOTSYS}/cmake/modules)
find_package(ROOT)

#---Checking for the existing installation of ROOT------------------------------------------------
message(STATUS "Configuring ${CMAKE_PROJECT_NAME} as standalone build ROOT package")
if(ROOT_FOUND)
  message(STATUS "Found ROOT installation at ${ROOTSYS}")
else()
  message(ERROR "ROOT installation not found")
endif()  


#---Minimal environment---------------------------------------------------------------------------
include(ROOTExports)
include(RootNewMacros)
include(CheckCompiler)
include(ROOTConfig)

#---Initialize project----------------------------------------------------------------------------
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#---Set Link and include directories--------------------------------------------------------------
set(CMAKE_INCLUDE_DIRECTORIES_BEFORE ON)
include_directories(${ROOT_INCLUDE_DIRS})
link_directories(${ROOT_LIBRARY_DIRS})

#---Configure and general additional files -------------------------------------------------------
#configure_file(${ROOTSYS}/cmake/modules/thisprojectsh.in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/this${CMAKE_PROJECT_NAME}.sh @ONLY)
#configure_file(${ROOTSYS}/cmake/modules/thisprojectcsh.in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/this${CMAKE_PROJECT_NAME}.csh @ONLY)
#install(FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/this${CMAKE_PROJECT_NAME}.sh
#              ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/this${CMAKE_PROJECT_NAME}.csh
#              PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ 
#              DESTINATION bin)



