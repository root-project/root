#-------------------------------------------------------------------------------
#
# CMakeLists.txt
#
# Top level CMakeLists.txt file for the roottest directory.
#
#-------------------------------------------------------------------------------

# Check the CMake version.
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# Set the name of the project.
project(roottest)

# Find ROOT.
find_package(ROOT QUIET)

if(NOT ROOT_FOUND)
  if(DEFINED ROOTSYS)
    if(EXISTS "${ROOTSYS}/etc/cmake/FindROOT.cmake")
      set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${ROOTSYS}/etc/cmake)
    else()
      message(FATAL_ERROR "Cannot find FindROOT.cmake!")
    endif()
  else()
    set(ROOTSYS $ENV{ROOTSYS})

    if(DEFINED ROOTSYS)
      if(EXISTS "${ROOTSYS}/etc/cmake/FindROOT.cmake")
        set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${ROOTSYS}/etc/cmake)
      else()
        message(FATAL_ERROR "Cannot find FindROOT.cmake!")
      endif()
    else()
      message(FATAL_ERROR "$ROOTSYS is not set, cannot find ROOT!")
    endif()
  endif()
endif()

# Setup environment.
set(ROOTTEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# Set the CMake module path. Here are all the custom CMake modules.
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${ROOTTEST_DIR}/cmake/modules")

# Include custom CMake modules.
include(CheckCompiler)
include(RootCTestMacros)

# Check, if roottest is built out of source.
ROOT_CHECK_OUT_OF_SOURCE_BUILD()

# Find GCCXML.
find_package(GCCXML REQUIRED)

# Detect bitness.
if(CMAKE_SYSTEM_PROCESSOR MATCHES amd64.*|x86_64.*)
  set(X86_64 1)
  message("-- Check for bitness: Found 64 bit architecture.")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES i686.*|i386.*|x86.*)
  set(X86 1)
  message("-- Check for bitness: Found 32 bit architecture.")
endif()

set(ENV{PATH} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}:$ENV{PATH}")
set(ENV{LD_LIBRARY_PATH} "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:$ENV{LD_LIBRARY_PATH}")
set(ENV{DYLD_LIBRARY_PATH} "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:$ENV{DYLD_LIBRARY_PATH}")
set(ENV{SHLIB_PATH} "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:$ENV{SHLIB_PATH}")
set(ENV{LIBPATH} "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:$ENV{LIBPATH}")
set(ENV{PYTHONPATH} "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:$ENV{PYTHONPATH}")
set(ENV{MANPATH} "${CMAKE_BINARY_DIR}/man/man1:$ENV{MANPATH}")

set(rootcint_program ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rootcint)

# Find python.
find_package(PythonInterp)
find_program(python_cmd ${PYTHON_EXECUTABLE})

# Enable and setup CTest. 
include(RootNewMacros)
include(RoottestCTest)
include(RootCTestMacros)
