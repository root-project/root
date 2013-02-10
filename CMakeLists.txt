#---Check if cmake has the required version-----------------------------------------------------
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
cmake_policy(SET CMP0005 NEW)
#---Set name of the project to "ROOT". Has to be done after check of cmake version--------------
project(ROOT)
set(IntegratedBuild ON)

#---Set pathes where to put the libraries, executables and headers------------------------------
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(ROOTSYS ${CMAKE_SOURCE_DIR})
set(HEADER_OUTPUT_PATH ${CMAKE_BINARY_DIR}/include)
set(ROOT_INCLUDE_DIR ${HEADER_OUTPUT_PATH})

#---Set the library version in the main CMakeLists.txt------------------------------------------
file(READ ${CMAKE_SOURCE_DIR}/build/version_number versionstr)
string(STRIP ${versionstr} versionstr)
string(REGEX REPLACE "([0-9]+)[.][0-9]+[/][0-9]+" "\\1" ROOT_MAJOR_VERSION ${versionstr})
string(REGEX REPLACE "[0-9]+[.]([0-9]+)[/][0-9]+" "\\1" ROOT_MINOR_VERSION ${versionstr})
string(REGEX REPLACE "[0-9]+[.][0-9]+[/]([0-9]+)" "\\1" ROOT_PATCH_VERSION ${versionstr})
set(ROOT_VERSION "${ROOT_MAJOR_VERSION}.${ROOT_MINOR_VERSION}.${ROOT_PATCH_VERSION}")

#---Where to look first for cmake modules, before ${CMAKE_ROOT}/Modules/ is checked-------------
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

#---Enable Folders in IDE like Visual Studio----------------------------------------------------
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#---Load some basic macros which are needed later for the confiuration and build----------------
include(RootNewMacros)
include(RootBuildOptions)
include(CheckCompiler)
include(MacroEnsureVersion)

#---Enable CTest package -----------------------------------------------------------------------
#include(CTest)
#---Check if the user wants to build the project in the source directory------------------------
ROOT_CHECK_OUT_OF_SOURCE_BUILD()

#---Here we look for installed software and switch on and of the different build options--------
include(SearchInstalledSoftware)
ROOT_SHOW_OPTIONS()

#---Recurse into the given subdirectories.  This does not actually cause another cmake executable
#  to run. The same process will walk through the project's entire directory structure.
add_subdirectory (cint)
add_subdirectory (core)
add_subdirectory (build)
add_subdirectory (math)
add_subdirectory (hist)
add_subdirectory (tree)
add_subdirectory (io)
add_subdirectory (net)
add_subdirectory (graf2d)
add_subdirectory (graf3d)
add_subdirectory (gui)
add_subdirectory (proof)
add_subdirectory (html)
add_subdirectory (montecarlo)
add_subdirectory (geom)
if(x11)
  add_subdirectory (rootx)
endif()
add_subdirectory (misc)
add_subdirectory (main)
add_subdirectory (bindings)
add_subdirectory (sql)
if(tmva)
  add_subdirectory(tmva)
endif()
if(roofit)
  add_subdirectory(roofit)
endif()

include(PostInstalledSoftware)

#---Configure and install various files neded later and for clients -----------------------------
include(RootConfiguration)

#install(EXPORT ${CMAKE_PROJECT_NAME}Exports DESTINATION cmake/modules) 

#---Installation of project-wise artifacts-------------------------------------------------------
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_INSTALL_PREFIX)
  install(FILES LICENSE DESTINATION .)
  install(DIRECTORY README/ DESTINATION README PATTERN ".svn" EXCLUDE)
  install(DIRECTORY etc/ DESTINATION etc USE_SOURCE_PERMISSIONS 
                         PATTERN ".svn" EXCLUDE 
                         REGEX system.rootrc EXCLUDE 
                         REGEX root.mimes EXCLUDE)
  install(DIRECTORY fonts/  DESTINATION fonts PATTERN ".svn" EXCLUDE)
  install(DIRECTORY icons/  DESTINATION icons PATTERN ".svn" EXCLUDE)
  install(DIRECTORY macros/ DESTINATION macros PATTERN ".svn" EXCLUDE)
  install(DIRECTORY man/    DESTINATION man PATTERN ".svn" EXCLUDE)
  install(DIRECTORY test/      DESTINATION test COMPONENT tests PATTERN ".svn" EXCLUDE)
  install(DIRECTORY tutorials/ DESTINATION tutorials COMPONENT tests PATTERN ".svn" EXCLUDE)
  install(DIRECTORY cmake/modules DESTINATION cmake PATTERN ".svn" EXCLUDE)
endif()

#---Configure Testing using CTest----------------------------------------------------------------
if(testing)
  include(RootCTest)
  add_subdirectory(test)
endif()

#---Packaging-------------------------------------------------------------------------------------
include(RootCPack)
