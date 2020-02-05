# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/unix package
############################################################################

set(Unix_dict_headers
  TUnixSystem.h
  PARENT_SCOPE
)

ROOT_OBJECT_LIBRARY(Unix
  src/TUnixSystem.cxx
)

target_include_directories(Unix PRIVATE ../clib/res ../foundation/res)

ROOT_INSTALL_HEADERS()
