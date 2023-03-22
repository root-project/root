# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/macosx package
############################################################################

if(NOT (APPLE AND cocoa))
  return()
endif()

set_property(TARGET Core APPEND PROPERTY DICT_HEADERS TMacOSXSystem.h)

target_include_directories(Core PRIVATE inc)

target_link_libraries(Core PRIVATE
  "-F/System/Library/PrivateFrameworks -framework CoreSymbolication"
)

target_link_libraries(Core PRIVATE "-framework Cocoa")

ROOT_OBJECT_LIBRARY(Macosx
  src/CocoaUtils.mm
  src/TMacOSXSystem.mm
)

target_compile_options(Macosx PRIVATE -ObjC++)
target_include_directories(Macosx PRIVATE ../unix/inc)

target_sources(Core PRIVATE $<TARGET_OBJECTS:Macosx>)

ROOT_INSTALL_HEADERS()
