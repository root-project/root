# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/clib package
############################################################################

set(Clib_dict_headers strlcpy.h
                      snprintf.h PARENT_SCOPE)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/res)

ROOT_OBJECT_LIBRARY(Clib *.c)

ROOT_INSTALL_HEADERS()
