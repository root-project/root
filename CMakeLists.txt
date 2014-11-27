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

#---Populate the configure arguments returned by 'root-config --config'-------------------------
get_cmake_property(variables CACHE_VARIABLES)
foreach(var ${variables})
  if((var MATCHES "_(LIBRARIES|LIBRARY|INCLUDE)") AND (NOT "${${var}}" STREQUAL "") AND
     (NOT "${var}" MATCHES "NOTFOUND"))
    if (var MATCHES "^QT_")
      # filter out the very long list of Qt libraries and include dirs
      if (var MATCHES "(QT_LIBRARY_DIR|QT_QTCORE_INCLUDE_DIR)")
        set(ROOT_CONFIGARGS "${ROOT_CONFIGARGS}${var}=${${var}} ")
      endif()
    else()
      if ((NOT var MATCHES "_(DOCS|TESTS|INSTALL)") AND (NOT var MATCHES "^_"))
        set(ROOT_CONFIGARGS "${ROOT_CONFIGARGS}${var}=${${var}} ")
      endif()
    endif()
  endif()
endforeach()

#---Move (copy) the headers and other directories to binary tree---------------------------------
add_custom_target(move_headers  COMMAND ${CMAKE_COMMAND} -DPREFIX=${CMAKE_BINARY_DIR} -DCOMPONENTS="headers"
                                        -P ${CMAKE_SOURCE_DIR}/cmake/scripts/local_install.cmake
                                COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/etc ${CMAKE_BINARY_DIR}/etc
                                COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/icons ${CMAKE_BINARY_DIR}/icons
                                COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/fonts ${CMAKE_BINARY_DIR}/fonts
                                COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/macros ${CMAKE_BINARY_DIR}/macros
                                COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/tutorials ${CMAKE_BINARY_DIR}/tutorials)
include_directories(${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR})

#---Recurse into the given subdirectories.  This does not actually cause another cmake executable
#  to run. The same process will walk through the project's entire directory structure.
add_subdirectory (interpreter)
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
get_property(__clingetcpch GLOBAL PROPERTY CLINGETCPCH)

add_custom_command(OUTPUT etc/dictpch/allLinkDefs.h
                          etc/dictpch/allHeaders.h
                          etc/dictpch/allCppflags.txt
                   COMMAND ${CMAKE_SOURCE_DIR}/build/unix/makepchinput.sh ${CMAKE_SOURCE_DIR} "." ${__clingetcpch}
                   DEPENDS ${CMAKE_SOURCE_DIR}/build/unix/makepchinput.sh ${__allFiles})

add_custom_command(OUTPUT etc/allDict.cxx.pch
                   COMMAND ${CMAKE_SOURCE_DIR}/etc/dictpch/makepch.sh etc/allDict.cxx.pch -I${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}
                   DEPENDS ${CMAKE_SOURCE_DIR}/etc/dictpch/makepch.sh
                           etc/dictpch/allLinkDefs.h
                           etc/dictpch/allHeaders.h
                           etc/dictpch/allCppflags.txt
                           rootcling)
add_custom_target(onepcm ALL DEPENDS etc/allDict.cxx.pch)
set_source_files_properties(${__allFiles} PROPERTIES GENERATED TRUE)
add_dependencies(onepcm ${__allTargets})
install(FILES ${CMAKE_BINARY_DIR}/etc/allDict.cxx.pch DESTINATION ${CMAKE_INSTALL_SYSCONFDIR})
install(DIRECTORY ${CMAKE_BINARY_DIR}/etc/dictpch DESTINATION ${CMAKE_INSTALL_SYSCONFDIR})

#---hsimple.root---------(use the executable for clearer dependencies and proper return code)---
if(NOT gnuinstall)   # TODO hsimple.root for gnuinstall
  add_custom_target(hsimple ALL DEPENDS tutorials/hsimple.root)
  add_dependencies(hsimple onepcm)
  ROOT_EXECUTABLE(hsimple.exe test/hsimple.cxx NOINSTALL LIBRARIES RIO Tree Hist)
  add_custom_command(OUTPUT tutorials/hsimple.root COMMAND hsimple.exe WORKING_DIRECTORY tutorials DEPENDS Cling)
  install(FILES ${CMAKE_BINARY_DIR}/tutorials/hsimple.root DESTINATION ${CMAKE_INSTALL_TUTDIR} COMPONENT tests)
endif()

#---version--------------------------------------------------------------------------------------
add_custom_target(version COMMAND ${CMAKE_SOURCE_DIR}/build/unix/makeversion.sh ${CMAKE_BINARY_DIR}
                          WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
#add_dependencies(version root.exe)

#---distribution commands------------------------------------------------------------------------
add_custom_target(distsrc COMMAND ${CMAKE_SOURCE_DIR}/build/unix/makedistsrc.sh
                          WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_custom_target(dist COMMAND cpack --config CPackConfig.cmake)

#---Configure and install various files neded later and for clients -----------------------------
include(RootConfiguration)

#---Installation of project-wise artifacts-------------------------------------------------------
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_INSTALL_PREFIX)
  install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DOCDIR})
  install(DIRECTORY README DESTINATION ${CMAKE_INSTALL_DOCDIR} PATTERN ".svn" EXCLUDE)
  install(DIRECTORY etc/ DESTINATION ${CMAKE_INSTALL_SYSCONFDIR} USE_SOURCE_PERMISSIONS
                         PATTERN ".svn" EXCLUDE
                         REGEX system.rootrc EXCLUDE
                         REGEX root.mimes EXCLUDE)
  install(DIRECTORY fonts/  DESTINATION ${CMAKE_INSTALL_FONTDIR} PATTERN ".svn" EXCLUDE)
  install(DIRECTORY icons/  DESTINATION ${CMAKE_INSTALL_ICONDIR} PATTERN ".svn" EXCLUDE)
  install(DIRECTORY macros/ DESTINATION ${CMAKE_INSTALL_MACRODIR} PATTERN ".svn" EXCLUDE)
  install(DIRECTORY man/    DESTINATION ${CMAKE_INSTALL_MANDIR} PATTERN ".svn" EXCLUDE)
  install(DIRECTORY test/      DESTINATION ${CMAKE_INSTALL_TESTDIR} COMPONENT tests PATTERN ".svn" EXCLUDE)
  install(DIRECTORY tutorials/ DESTINATION ${CMAKE_INSTALL_TUTDIR} COMPONENT tests PATTERN ".svn" EXCLUDE)
  install(DIRECTORY cmake/modules DESTINATION ${CMAKE_INSTALL_CMAKEDIR} PATTERN ".svn" EXCLUDE)
  install(FILES build/misc/root.m4 DESTINATION ${CMAKE_INSTALL_ACLOCALDIR})
endif()

#---Configure Testing using CTest----------------------------------------------------------------
configure_file(${CMAKE_SOURCE_DIR}/cmake/modules/CTestCustom.cmake ${CMAKE_BINARY_DIR} COPYONLY)
if(testing)
  if(gnuinstall)
    message(FATAL_ERROR "You cannot enable to run the tests in the build directory (testing=ON) and at the same time select a GNU style installation (gnuinstall=ON). This is because the code is built with the final installation location (CMAKE_INSTALL_PREFIX) for many ancillary files (tutorials, etc, icons, macros, etc.) and these are not available before installation. Tests can be run against the installation once is completed. See the instructions in the web")
  endif()
  include(RootCTest)
  if(roottest)
    find_package(Git REQUIRED)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTPUT_VARIABLE GIT_BRANCH OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(GIT_BRANCH STREQUAL HEAD)
      string(REPLACE "origin/" "" GIT_BRANCH "$ENV{GIT_BRANCH}")  # Comming from Jenkins checkout
    endif()
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/roottest)
      execute_process(COMMAND ${GIT_EXECUTABLE} clean -xfdq WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/roottest)
      execute_process(COMMAND ${GIT_EXECUTABLE} checkout ${GIT_BRANCH} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/roottest)
      execute_process(COMMAND ${GIT_EXECUTABLE} fetch origin WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/roottest)
      execute_process(COMMAND ${GIT_EXECUTABLE} rebase origin/${GIT_BRANCH} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/roottest)
      file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/roottest)
      add_subdirectory(roottest)
    elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../roottest)
      add_subdirectory(../roottest roottest)
    else()
      message("-- Could not find roottest directory! Cloning from the repository...")
      execute_process(COMMAND ${GIT_EXECUTABLE} clone -b ${GIT_BRANCH} http://root.cern.ch/git/roottest.git
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
      add_subdirectory(roottest)
    endif()
  endif()
endif()

#---Packaging-------------------------------------------------------------------------------------
include(RootCPack)
