# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# The vendored cppjit tree builds through ROOT-owned CMake files inside it
# (src/, python/, test/); upstream's own top-level CMakeLists.txt is vendored
# but unreferenced.
add_subdirectory(cppjit/src)
add_subdirectory(cppjit/python)
ROOT_ADD_TEST_SUBDIRECTORY(cppjit/test)

add_subdirectory(pythonizations)
