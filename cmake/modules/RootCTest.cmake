# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

#---------------------------------------------------------------------------------------------------
#  RootCTest.cmake
#   - basic setup for testing ROOT using CTest
#---------------------------------------------------------------------------------------------------

#---Deduce the build name--------------------------------------------------------
set(BUILDNAME ${ROOT_ARCHTECTURE}-${CMAKE_BUILD_TYPE})

enable_testing()
include(CTest)

#---A number of operations to allow running the tests from the build directory-----------------------
set(ROOT_DIR ${CMAKE_BINARY_DIR})

#---Test products should not be poluting the standard destinations--------------------------------
unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
unset(CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY)

if(WIN32)
  foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG})
    unset(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG})
    unset(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG})
  endforeach()
endif()

#---Add all subdirectories with tests-----------------------------------------------------------

get_property(test_dirs GLOBAL PROPERTY ROOT_TEST_SUBDIRS)
foreach(d ${test_dirs})
  list(APPEND test_list ${d})
endforeach()

if(test_list)
  list(SORT test_list)
endif()

foreach(d ${test_list})
  if(d STREQUAL tutorials)
    add_subdirectory(${d} runtutorials)  # to avoid clashes with the tutorial sources copied to binary tree
  else()
    add_subdirectory(${d})
  endif()
endforeach()

# When ninja is in use, tests that compile an executable might try to rebuild the entire build tree.
# If multiple of these are invoked in parallel, ninja will suffer from race conditions.
# To solve this, do the following:
# - Add a test that updates the build tree (equivalent to "ninja all"). This one will run in complete isolation.
# - Make all tests that require a ninja build depend on the above test.
# - Use a RESOURCE_LOCK on all tests that invoke ninja, so no two tests will invoke ninja in parallel
if(CMAKE_GENERATOR MATCHES Ninja AND NOT MSVC)
  add_test(NAME ninja-build-all
      COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR})
  set_tests_properties(ninja-build-all PROPERTIES
      RESOURCE_LOCK NINJA_BUILD
      FIXTURES_SETUP NINJA_BUILD_ALL
      RUN_SERIAL True)
endif()
