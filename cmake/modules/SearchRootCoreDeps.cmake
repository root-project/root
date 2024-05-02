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
  - PYTHON_VERSION_MAJOR
  - PYTHON_VERSION_MINOR

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
find_package(Python3 3.8 COMPONENTS Interpreter Development NumPy)
if(Python3_Interpreter_FOUND)
  set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
  set(PYTHON_VERSION_STRING "${Python3_VERSION}" CACHE INTERNAL "" FORCE)
  set(PYTHON_VERSION_MAJOR "${Python3_VERSION_MAJOR}" CACHE INTERNAL "" FORCE)
  set(PYTHON_VERSION_MINOR "${Python3_VERSION_MINOR}" CACHE INTERNAL "" FORCE )
endif()

# if development parts not found, one still need python executable to run some scripts
if(NOT PYTHON_EXECUTABLE)
  find_package(Python3)
  if(Python3_FOUND)
      message(STATUS "Found python3 executable ${Python3_EXECUTABLE}, required only for ROOT compilation.")
      set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
  endif()
endif()
