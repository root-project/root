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

target_include_directories(Unix PRIVATE 
   ${PROJECT_SOURCE_DIR}/core/clib/res
   ${PROJECT_SOURCE_DIR}/core/foundation/res
   ${PROJECT_SOURCE_DIR}/core/base/inc
   ${PROJECT_SOURCE_DIR}/core/clib/inc
   ${PROJECT_SOURCE_DIR}/core/meta/inc
   ${PROJECT_SOURCE_DIR}/core/cont/inc
   ${PROJECT_SOURCE_DIR}/core/thread/inc
   ${PROJECT_SOURCE_DIR}/core/textinput/inc
   ${PROJECT_SOURCE_DIR}/core/gui/inc
   ${PROJECT_SOURCE_DIR}/core/foundation/inc
   ${PROJECT_BINARY_DIR}/ginclude
)


ROOT_INSTALL_HEADERS()
