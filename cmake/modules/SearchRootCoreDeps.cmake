# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

#[[
The purpose of this machinery is to search Python and set both the ROOT and PyROOT related
Python variables.

Variables set when Interpreter is found:
  - PYTHON_EXECUTABLE
  - PYTHON_VERSION_STRING
  - PYTHON_UNDER_VERSION_STRING: Python version with "_" replacing ".". Used to give a version-dependent name to the libraries, to allow
  multiple builds
  - PYTHON_VERSION_MAJOR
  - PYTHON_VERSION_MINOR

Variables set when Development is found:
  - PYTHON_INCLUDE_DIRS
  - PYTHON_LIBRARIES
  - PYTHON_LINK_OPTIONS: necessary on MacOS to link to the XCode Python

^^^^^^^^

Explanation of the machinery:

The first distinction is based on the CMake version used to build. If it is >= 3.14, than PyROOT can be built
for multiple Python versions. In case CMake >= 3.14, then we check if PYTHON_EXECUTABLE was specified by the user;
if so, PyROOT is built with only that version.

If PYTHON_EXECUTABLE is specified:
    - we check which version it is (from here call X) and call the relative
    find_package(PythonX COMPONENTS Interpreter Development Numpy)
    - if Interpreter is found, we set the ROOT related Python variables
    - if Development is found, we set the PyROOT variables (+ Numpy ones)

If PYTHON_EXECUTABLE is NOT specified:
    - we look for Python3, since we want the highest version to be the preferred one
    - if Python3 Interpreter is found, we set the ROOT related Python variables
    - if Python3 Development is found, we set the PyROOT_Main variables (+ Numpy ones)
]]

message(STATUS "Looking for Python")

# On macOS, prefer user-provided Pythons.
set(Python3_FIND_FRAMEWORK LAST)

# - Look for Python3 and set the deprecated variables to the ones set
# automatically by find_package(Python3 ...)
find_package(Python3 COMPONENTS Interpreter Development NumPy)
if(Python3_Interpreter_FOUND)
  set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
  set(PYTHON_VERSION_STRING "${Python3_VERSION}" CACHE INTERNAL "" FORCE)
  set(PYTHON_VERSION_MAJOR "${Python3_VERSION_MAJOR}" CACHE INTERNAL "" FORCE)
  set(PYTHON_VERSION_MINOR "${Python3_VERSION_MINOR}" CACHE INTERNAL "" FORCE )
  set(PYTHON_UNDER_VERSION_STRING "${Python3_VERSION_MAJOR}_${Python3_VERSION_MINOR}" CACHE INTERNAL "" FORCE )
  if(Python3_Development_FOUND)
    set(PYTHON_INCLUDE_DIRS "${Python3_INCLUDE_DIRS}" CACHE INTERNAL "" FORCE)
    set(PYTHON_LIBRARIES "${Python3_LIBRARIES}" CACHE INTERNAL "" FORCE)
    set(PYTHON_LIBRARY_DIR "${Python3_LIBRARY_DIRS}" CACHE INTERNAL "" FORCE)
    set(PYTHON_LINK_OPTIONS "${Python3_LINK_OPTIONS}" CACHE INTERNAL "" FORCE)
  endif()
  if(Python3_NumPy_FOUND)
    set(NUMPY_FOUND ${Python3_NumPy_FOUND})
    set(NUMPY_INCLUDE_DIRS "${Python3_NumPy_INCLUDE_DIRS}" CACHE INTERNAL "" FORCE)
  endif()
endif()

if(NOT Python3_Development_FOUND)
  message(WARNING "No Python 3 development packages were found; PyROOT will not be built.")
endif()

# if development parts not found, one still need python executable to run some scripts
if(NOT PYTHON_EXECUTABLE)
  find_package(Python3)
  if(Python3_FOUND)
      message(STATUS "Found python3 executable ${Python3_EXECUTABLE}, required only for ROOT compilation.")
      set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
  endif()
endif()
