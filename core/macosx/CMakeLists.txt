# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/macosx package
############################################################################

set(Macosx_dict_headers
  TMacOSXSystem.h
  PARENT_SCOPE
)

set(core_tgts ${core_tgts} Macosx PARENT_SCOPE)

ROOT_OBJECT_LIBRARY(Macosx
  src/CocoaUtils.mm
  src/TMacOSXSystem.mm
)

target_compile_options(Macosx PRIVATE -ObjC++)

target_include_directories(Macosx PRIVATE ${CMAKE_SOURCE_DIR}/core/unix/inc)

ROOT_INSTALL_HEADERS()
