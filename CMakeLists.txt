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
find_package(ROOT QUIET COMPONENTS Core RIO Net Hist Gpad Tree Rint Matrix MathCore)

if(NOT ROOT_FOUND)

  # ROOTSYS specified as a cmake -D define?
  if(DEFINED ROOTSYS)
    if(EXISTS "${ROOTSYS}/etc/cmake/FindROOT.cmake")
      set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${ROOTSYS}/etc/cmake)
      find_package(ROOT PATHS ${ROOTSYS}/etc/cmake/
                   COMPONENTS Core RIO Net Hist Gpad Tree Rint Matrix MathCore)

    else()
      message(FATAL_ERROR "Cannot find FindROOT.cmake!")
    endif()
  else()
    # Try to retrieve the ROOTSYS path from the current environment.
    set(ROOTSYS $ENV{ROOTSYS})

    if(DEFINED ROOTSYS)
      if(EXISTS "${ROOTSYS}/etc/cmake/FindROOT.cmake")
        set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${ROOTSYS}/etc/cmake)
        find_package(ROOT PATHS ${ROOTSYS}/etc/cmake/
                     COMPONENTS Core RIO Net Hist Gpad Tree Rint Matrix MathCore)
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
set(ROOTTEST_ENV_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(ROOTTEST_ENV_PYTHONPATH ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(ROOTTEST_ENV_LIBRARYPATH ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

# Resolve symbolic links for the ROOTTEST_DIR variable.
get_filename_component(ROOTTEST_DIR ${CMAKE_CURRENT_SOURCE_DIR} ABSOLUTE)

set(rootcint_program ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rootcint)

# Set the CMake module path. Here are all the custom CMake modules.
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${ROOTTEST_DIR}/cmake/modules")

# Include custom CMake modules.
include(CheckCompiler)
include(RootCTestMacros)

# Find GCCXML.
find_package(GCCXML REQUIRED)

# Find python.
find_package(PythonInterp)
find_program(python_cmd ${PYTHON_EXECUTABLE})

# Setup standard includes / links.
include_directories(${ROOT_INCLUDE_DIRS} ${ROOT_INCLUDE_DIR})
link_directories(${ROOT_LIBRARY_DIR})

# Enable and setup CTest.
include(RootNewMacros)
include(RoottestCTest)
include(RootCTestMacros)

message("-- Scanning subdirectories for tests...")
ROOTTEST_FIND_TESTDIRS()

message("ROOT LIBS: ${ROOT_LIBRARIES}")
