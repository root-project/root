#-------------------------------------------------------------------------------
#
# CMakeLists.txt
#
# Top level CMakeLists.txt file for the roottest directory.
#
#-------------------------------------------------------------------------------

# Check the CMake version.
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

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

# You need to tell CMake where to find the ROOT installation. This can be done in a number of ways:
#   - ROOT built with classic configure/make use the provided $ROOTSYS/etc/cmake/FindROOT.cmake
#   - ROOT built with CMake. Add in CMAKE_PREFIX_PATH the installation prefix for ROOT
list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS})

# Check, if ROOTSYS is set.
if(NOT DEFINED ROOTSYS)
  set(ROOTSYS $ENV{ROOTSYS})
  if(NOT DEFINED ROOTSYS)
    message(FATAL_ERROR "ROOTSYS is not set!")
  endif()
endif()

# If ROOT_SOURCE_DIR is set, roottest is built as part of the root build.
# If ROOT_SOURCE_DIR is not set, search for an installation.
if(DEFINED ROOT_SOURCE_DIR)
  find_package(ROOT REQUIRED)
  get_filename_component(ROOT_INCLUDE_DIR "${ROOTSYS}/include" ABSOLUTE)
  set(ROOT_INCLUDE_DIRS ${ROOT_INCLUDE_DIR})
  set(ROOT_LIBRARIES Core RIO Net Hist Gpad Tree Rint Matrix MathCore)
  set(ROOT_genreflex_CMD ${ROOTSYS}/bin/genreflex)
  set(ROOT_rootcint_CMD ${ROOTSYS}/bin/rootcint)
  set(ROOT_rootcling_CMD rootcling)
else()# testing using an installation
  find_package(ROOT REQUIRED COMPONENTS RIO Net Hist Gpad Tree Rint Matrix MathCore)
  include(${ROOT_USE_FILE})
  if(DEFINED ROOT_CONFIG_EXECUTABLE) #---If ROOT was built with the classic configure/make---
    find_program(ROOT_root_CMD root.exe PATHS ${ROOTSYS}/bin)
    find_program(ROOT_genreflex_CMD genreflex PATHS ${ROOTSYS}/bin)
    find_program(ROOT_rootcint_CMD rootcint PATHS ${ROOTSYS}/bin)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules)
    include(RootNewMacros)
  endif()
  enable_testing()
endif()

get_filename_component(ROOT_BINARY_DIR  "${ROOTSYS}/bin" ABSOLUTE)
get_filename_component(ROOT_LIBRARY_DIR "${ROOTSYS}/lib" ABSOLUTE)

# Detect bitness.
if(CMAKE_SYSTEM_PROCESSOR MATCHES amd64.*|x86_64.*|aarch64.*|ppc64.*
   OR (CMAKE_VERSION VERSION_LESS 3.0 AND CMAKE_SYSTEM_NAME STREQUAL Darwin) )
  set(64BIT 1)
  message("-- Check for bitness: Found 64 bit architecture.")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES i686.*|i386.*|x86.*)
  set(32BIT 1)
  message("-- Check for bitness: Found 32 bit architecture.")
endif()

# Setup environment.
set(ROOTTEST_ENV_PATH ${ROOT_BINARY_DIR})
set(ROOTTEST_ENV_PYTHONPATH ${ROOT_LIBRARY_DIR})
set(ROOTTEST_ENV_LIBRARYPATH ${ROOT_LIBRARY_DIR})

# Resolve symbolic links for the ROOTTEST_DIR variable.
get_filename_component(ROOTTEST_DIR ${CMAKE_CURRENT_SOURCE_DIR} ABSOLUTE)

# Set some variables that customizes the behaviour of the ROOT macros
set(CMAKE_ROOTTEST_DICT ON)

# Set the CMake module path. Here are all the custom CMake modules.
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${ROOTTEST_DIR}/cmake/modules")

# Find python.
if(ROOT_python_FOUND)
  find_package(PythonInterp)
  find_package(PythonLibs)
endif()

# Setup standard includes / links.
include_directories(${ROOT_INCLUDE_DIRS} ${ROOT_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${ROOT_LIBRARY_DIR})

# Enable and setup CTest.
include(RoottestCTest)
include(RootCTestMacros)

message("-- Scanning subdirectories for tests...")
ROOTTEST_ADD_TESTDIRS()
