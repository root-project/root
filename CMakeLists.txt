#---Check if cmake has the required version-----------------------------------------------------
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
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

#---Where to look first for cmake modules, before ${CMAKE_ROOT}/Modules/ is checked-------------
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

#---Enable Folders in IDE like Visual Studio----------------------------------------------------
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#---Load some basic macros which are needed later for the confiuration and build----------------
include(RootNewMacros)
include(CheckCompiler)
include(MacroEnsureVersion)

#---Enable CTest package -----------------------------------------------------------------------
#include(CTest)
#---Check if the user wants to build the project in the source directory------------------------
ROOT_CHECK_OUT_OF_SOURCE_BUILD()

#---Here we declare the required build options--------------------------------------------------
include(RootBuildOptions)
#---Here we look for installed software and switch on and of the different build options--------
include(SearchInstalledSoftware)
ROOT_SHOW_OPTIONS()

#---Set the library version in the main CMakeLists.txt------------------------------------------
file(READ ${CMAKE_SOURCE_DIR}/build/version_number versionstr)
string(STRIP ${versionstr} versionstr)
string(REGEX REPLACE "([0-9]+)[.][0-9]+[/][0-9]+" "\\1" ROOT_MAJOR_VERSION ${versionstr})
string(REGEX REPLACE "[0-9]+[.]([0-9]+)[/][0-9]+" "\\1" ROOT_MINOR_VERSION ${versionstr})
string(REGEX REPLACE "[0-9]+[.][0-9]+[/]([0-9]+)" "\\1" ROOT_PATCH_VERSION ${versionstr})
set(ROOT_VERSION "${ROOT_MAJOR_VERSION}.${ROOT_MINOR_VERSION}.${ROOT_PATCH_VERSION}")

#---Configure and install various files neded later and for clients -----------------------------
include(RootConfiguration)

#---Add dummy tatget to trigger the build of tests libraries and programs-------------------------
add_custom_target(check)

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
if(NOT WIN32)
  add_Subdirectory (rootx)
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
install(EXPORT ${CMAKE_PROJECT_NAME}Exports DESTINATION cmake/modules) 

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

#---Packaging-------------------------------------------------------------------------------------
include(InstallRequiredSystemLibraries)
configure_file(cmake/modules/CMakeCPackOptions.cmake.in CMakeCPackOptions.cmake @ONLY)
configure_file(README/README README.txt COPYONLY)
configure_file(LICENSE LICENSE.txt COPYONLY)
set(CPACK_PROJECT_CONFIG_FILE ${CMAKE_BINARY_DIR}/CMakeCPackOptions.cmake)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "ROOT project")
set(CPACK_PACKAGE_VENDOR "HEPSoft")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_BINARY_DIR}/README.txt")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_BINARY_DIR}/LICENSE.txt")
set(CPACK_RESOURCE_FILE_README "${CMAKE_BINARY_DIR}/README.txt")
set(CPACK_PACKAGE_VERSION_MAJOR ${ROOT_MAJOR_VERSION})
set(CPACK_PACKAGE_VERSION_MINOR ${ROOT_MINOR_VERSION})
set(CPACK_PACKAGE_VERSION_PATCH ${ROOT_PATCH_VERSION})
set(CPACK_PACKAGE_RELOCATABLE True)
set(CPACK_PACKAGE_INSTALL_DIRECTORY "ROOT ${ROOT_MAJOR_VERSION}.${ROOT_MINOR_VERSION}")
if(CMAKE_BUILD_TYPE STREQUAL Release)
  set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${ROOT_VERSION}-${ROOT_ARCHITECTURE}")
else()
  set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${ROOT_VERSION}-${ROOT_ARCHITECTURE}-${CMAKE_BUILD_TYPE}")
endif()
set(CPACK_SOURCE_STRIP_FILES "")
set(CPACK_PACKAGE_EXECUTABLES "root" "ROOT")
include(CPack)

#---Define components and installation types--------------------------------------------------
cpack_add_install_type(full      DISPLAY_NAME "Full Installation")
cpack_add_install_type(minimal   DISPLAY_NAME "Minimal Installation")
cpack_add_install_type(developer DISPLAY_NAME "Developer Installation")
cpack_add_component(applications DISPLAY_NAME "ROOT Applications" 
                                 DESCRIPTION "ROOT executables such as root.exe"
								 INSTALL_TYPES full minimal developer)
cpack_add_component(libraries DISPLAY_NAME "ROOT Libraries" 
                                 DESCRIPTION "All ROOT libraries and dictionaries"
								 INSTALL_TYPES full minimal developer)
cpack_add_component(headers DISPLAY_NAME "C++ Headers" 
                                 DESCRIPTION "These are needed to do any development"
								 INSTALL_TYPES full developer)
cpack_add_component(tests DISPLAY_NAME "ROOT Tests and Tutorials" 
                                 DESCRIPTION "These are needed to do any test and tutorial"
								 INSTALL_TYPES full developer)
								 
								 
