#---Check if cmake has the required version-----------------------------------------------------
cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)
cmake_policy(SET CMP0005 NEW)
#---Set name of the project to "ROOT". Has to be done after check of cmake version--------------
project(ROOT)
set(IntegratedBuild ON)

#---Set pathes where to put the libraries, executables and headers------------------------------
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(ROOTSYS ${CMAKE_BINARY_DIR})
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
include(RootBuildOptions)
include(RootNewMacros)
include(CheckCompiler)
include(MacroEnsureVersion)

#---Enable CTest package -----------------------------------------------------------------------
#include(CTest)
if(testing)
  enable_testing()
endif()

#---Check if the user wants to build the project in the source directory------------------------
ROOT_CHECK_OUT_OF_SOURCE_BUILD()

#---Here we look for installed software and switch on and of the different build options--------
include(SearchInstalledSoftware)
ROOT_SHOW_OPTIONS()

#---Recurse into the given subdirectories.  This does not actually cause another cmake executable
#  to run. The same process will walk through the project's entire directory structure.
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
add_subdirectory (rootx)
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

ROOT_ADD_TEST_SUBDIRECTORY(test)
ROOT_ADD_TEST_SUBDIRECTORY(tutorials)

#---Global PCH-----------------------------------------------------------------------------------
get_property(__allTargets GLOBAL PROPERTY ROOT_DICTIONARY_TARGETS)
get_property(__allFiles GLOBAL PROPERTY ROOT_DICTIONARY_FILES)
add_custom_command(OUTPUT include/allLinkDef.h include/allHeaders.h cppflags.txt
                   COMMAND ${CMAKE_SOURCE_DIR}/build/unix/makeonepcm.sh ${CMAKE_SOURCE_DIR} '*'
                   DEPENDS ${CMAKE_SOURCE_DIR}/build/unix/makeonepcm.sh ${__allFiles})
set(__cxxflags -D__CLING__ -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -DLDAP_DEPRECATED -DQT3_SUPPORT -Wno-deprecated 
               -Iinclude -Ietc -Ietc/cling -I${CMAKE_SOURCE_DIR})
add_custom_command(OUTPUT etc/allDict.cxx.pch
                  COMMAND bin/rootcling_tmp -1 -f etc/allDict.cxx -noDictSelection -c ${__cxxflags} `cat cppflags.txt | sort | uniq` include/allHeaders.h include/allLinkDef.h
                  COMMAND ${CMAKE_COMMAND} -E copy etc/allDict_rdict.pch etc/allDict.cxx.pch
                  DEPENDS include/allLinkDef.h include/allHeaders.h rootcling_tmp)
add_custom_target(onepcm ALL DEPENDS etc/allDict.cxx.pch)
set_source_files_properties(${__allFiles} PROPERTIES GENERATED TRUE)
add_dependencies(onepcm ${__allTargets})
install(FILES ${CMAKE_BINARY_DIR}/etc/allDict.cxx.pch DESTINATION ${CMAKE_INSTALL_SYSCONFDIR})

#------------------------------------------------------------------------------------------------

include(PostInstalledSoftware)

#---Configure and install various files neded later and for clients -----------------------------
include(RootConfiguration)

#---Installation of project-wise artifacts-------------------------------------------------------
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_INSTALL_PREFIX)
  install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DOCDIR})
  install(DIRECTORY README/ DESTINATION ${CMAKE_INSTALL_DOCDIR} PATTERN ".svn" EXCLUDE)
  install(DIRECTORY etc/ DESTINATION ${CMAKE_INSTALL_SYSCONFDIR} USE_SOURCE_PERMISSIONS
                         PATTERN ".svn" EXCLUDE 
                         REGEX system.rootrc EXCLUDE 
                         REGEX root.mimes EXCLUDE)
  install(DIRECTORY fonts/  DESTINATION ${CMAKE_INSTALL_FONTDIR} PATTERN ".svn" EXCLUDE)
  install(DIRECTORY icons/  DESTINATION ${CMAKE_INSTALL_ICONDIR} PATTERN ".svn" EXCLUDE)
  install(DIRECTORY macros/ DESTINATION ${CMAKE_INSTALL_MACRODIR} PATTERN ".svn" EXCLUDE)
  install(DIRECTORY man/    DESTINATION ${CMAKE_INSTALL_MANDIR} PATTERN ".svn" EXCLUDE)
  install(DIRECTORY test/      DESTINATION ${CMAKE_INSTALL_TESTDIR} COMPONENT tests PATTERN ".svn" EXCLUDE)
  install(DIRECTORY tutorials/ DESTINATION ${CMAKE_INSTALL_TUTDIR} COMPONENT tutorials PATTERN ".svn" EXCLUDE)
  install(DIRECTORY cmake/modules DESTINATION ${CMAKE_INSTALL_CMAKEDIR} PATTERN ".svn" EXCLUDE)
  install(FILES build/misc/root.m4 DESTINATION ${CMAKE_INSTALL_ACLOCALDIR})
endif()

#---Configure Testing using CTest----------------------------------------------------------------
if(testing)
  include(RootCTest)
  if(roottest)
    find_package(Git REQUIRED)
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/roottest)
      execute_process(COMMAND ${GIT_EXECUTABLE} clean -xfq WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/roottest)
      execute_process(COMMAND ${GIT_EXECUTABLE} pull WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/roottest)
      add_subdirectory(roottest)
    elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../roottest)
      add_subdirectory(../roottest roottest)
    else()
      message("-- Could not find roottest directory! Cloning from the repository...")
      execute_process(COMMAND ${GIT_EXECUTABLE} clone http://root.cern.ch/git/roottest.git
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
      add_subdirectory(roottest)
    endif()
  endif()
endif()

#---Packaging-------------------------------------------------------------------------------------
include(RootCPack)
