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

# Check, if roottest is built out of source.
string(COMPARE EQUAL ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} insource)
if(insource)
  file(REMOVE_RECURSE ${CMAKE_SOURCE_DIR}/Testing)
  file(REMOVE ${CMAKE_SOURCE_DIR}/DartConfiguration.tcl)
  message(FATAL_ERROR "roottest should be installed as an out of source build, "
                      "to keep the source directory clean. Please create a "
                      "extra build directory and run the command 'cmake "
                      "<path_to_source_dir>' in this newly created directory. "
                      "You have also to delete the directory CMakeFiles and "
                      "the file CMakeCache.txt in the source directory. "
                      "Otherwise cmake will complain even if you run it from "
                      "an out-of-source directory.")
endif()

# Find ROOT.
find_package(ROOT QUIET)

if(NOT ROOT_FOUND)

  # ROOTSYS specified as a cmake -D define?
  if(DEFINED ROOTSYS)
    if(EXISTS "${ROOTSYS}/etc/cmake/FindROOT.cmake")
      set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${ROOTSYS}/etc/cmake)
      find_package(ROOT PATHS ${ROOTSYS}/etc/cmake/)
    else()
      message(FATAL_ERROR "Cannot find FindROOT.cmake!")
    endif()
  else()
    # Try to retrieve the ROOTSYS path from the current environment.
    set(ROOTSYS $ENV{ROOTSYS})

    if(DEFINED ROOTSYS)
      if(EXISTS "${ROOTSYS}/etc/cmake/FindROOT.cmake")
        set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${ROOTSYS}/etc/cmake)
        find_package(ROOT PATHS ${ROOTSYS}/etc/cmake/)
      else()
        message(FATAL_ERROR "Cannot find FindROOT.cmake!")
      endif()
    else()
      message(FATAL_ERROR "$ROOTSYS is not set, cannot find ROOT!")
    endif()
  endif()

  get_filename_component(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${ROOTSYS}/bin" ABSOLUTE)
  get_filename_component(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${ROOTSYS}/lib" ABSOLUTE)
else()
  if(NOT DEFINED ROOTSYS)
    set(ROOTSYS $ENV{ROOTSYS})
    get_filename_component(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${ROOTSYS}/bin" ABSOLUTE)
    get_filename_component(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${ROOTSYS}/lib" ABSOLUTE)
  endif()
endif()

# Detect bitness.
if(CMAKE_SYSTEM_PROCESSOR MATCHES amd64.*|x86_64.*)
  set(X86_64 1)
  message("-- Check for bitness: Found 64 bit architecture.")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES i686.*|i386.*|x86.*)
  set(X86 1)
  message("-- Check for bitness: Found 32 bit architecture.")
endif()

# Setup environment.
set(ROOTTEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(rootcint_program ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rootcint)

# Set the CMake module path. Here are all the custom CMake modules.
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${ROOTTEST_DIR}/cmake/modules")

# Include custom CMake modules.
include(CheckCompiler)
include(RootCTestMacros)

# Find GCCXML.
find_package(GCCXML REQUIRED)

if(WIN32)
  set(ENV{PATH} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY};$ENV{PATH}")
endif()

if(CMAKE_SYSTEM_NAME MATCHES Linux)
  set(ENV{LD_LIBRARY_PATH} "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:$ENV{LD_LIBRARY_PATH}")
elseif(APPLE)
  set(ENV{DYLD_LIBRARY_PATH} "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:$ENV{DYLD_LIBRARY_PATH}")
endif()

set(ENV{PYTHONPATH} "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:$ENV{PYTHONPATH}")

# Set of libraries that are used as standard dependencies. Dependencies needed
# so make -j does not fail when compiling macros, but the libraries do not exist
# yet.
set(ROOTTEST_LIB_DEPENDS Core RIO Net Hist Gpad Tree Rint Matrix Mathcore)

# Find python.
find_package(PythonInterp)
find_program(python_cmd ${PYTHON_EXECUTABLE})

# Enable and setup CTest.
include(RootNewMacros)
include(RoottestCTest)
include(RootCTestMacros)
