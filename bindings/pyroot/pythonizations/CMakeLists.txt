# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

################################################################
# CMakeLists.txt file for building ROOT pythonizations
################################################################

# Create meta-target PyROOT3 (INTERFACE library)
# Export of targets are not supported for custom targets(add_custom_targets())
add_library(PyROOT INTERFACE)
target_link_libraries(PyROOT INTERFACE cppjit_backend cppjit)

# For the ROOT Python package
add_subdirectory(python)

ROOT_ADD_TEST_SUBDIRECTORY(test)
