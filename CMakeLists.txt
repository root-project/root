# CMakeLists.txt for the ROOT test programs.
# This shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Author: Pere Mato, 20/12/2011
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(roottest)

find_package(ROOT REQUIRED)
include(RootNewMacros)
include(CheckCompiler)
include(RootCTest)

if(NOT DEFINED ROOT_SOURCE_DIR)
  message(WARNING "ROOT_SOURCE_DIR needs to be defined")
  return()
endif()

find_package(PythonInterp)

#include(${ROOT_USE_FILE})
include_directories(${ROOT_INCLUDE_DIRS})

find_program(diff_cmd diff)
find_program(cat_cmd cat)
find_program(grep_cmd grep)
find_program(python_cmd ${PYTHON_EXECUTABLE})


function(ROOT_ADD_ROOTTEST test)
  PARSE_ARGUMENTS(ARG "MACRO;OUTREF;OUTCNV;DEPENDS" "DEBUG" ${ARGN})
  if(ARG_MACRO MATCHES "[.]C")
    set(command root -q -l -b)
  elseif(ARG_MACRO MATCHES "[.]py")
    set(command ${python_cmd})
  endif()
  if(ARG_DEPENDS)
    set(depends DEPENDS ${ARG_DEPENDS})
  endif()
  ROOT_ADD_TEST(${test}-run COMMAND ${command} ${ARG_MACRO}
                            POSTCMD ${cat_cmd} ${test}.log 
                            OUTPUT ${test}.log
                            WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                            ${depends})
  if(ARG_OUTREF)
    if(ARG_OUTCNV)
      ROOT_ADD_TEST(${test}-diff COMMAND ${ARG_OUTCNV} ${test}.log
                                 POSTCMD ${diff_cmd} -u -b ${test}.cnv.log ${ARG_OUTREF}
                                 OUTPUT ${test}.cnv.log
                                 WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                                 DEPENDS ${test}-run)
      
    else()
      ROOT_ADD_TEST(${test}-diff COMMAND ${diff_cmd} -u -b ${test}.log ${ARG_OUTREF}
                                 WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                                 DEPENDS ${test}-run)
    endif()                           
  endif()
endfunction()

#---Ssan all directories to look for tests-----------------------------------------------------
file(GLOB_RECURSE files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CMakeLists.txt)
foreach( file ${files} )
  get_filename_component(path ${file} PATH)
  if(path)
    add_subdirectory(${path})
  endif()
endforeach()
