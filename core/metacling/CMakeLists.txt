# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/metacling package
############################################################################

add_subdirectory(src)

set(core_tgts ${core_tgts} MetaCling PARENT_SCOPE)

ROOT_ADD_TEST_SUBDIRECTORY(test)
