# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/macosx package
############################################################################

if(NOT APPLE)
  return()
endif()

target_link_libraries(Core PRIVATE
  "-F/System/Library/PrivateFrameworks -framework CoreSymbolication"
)

if(cocoa)
  set_property(TARGET Core APPEND PROPERTY DICT_HEADERS TMacOSXSystem.h)

  target_include_directories(Core PRIVATE inc)

  target_link_libraries(Core PRIVATE "-framework Cocoa")

  ROOT_OBJECT_LIBRARY(Macosx
    src/CocoaUtils.mm
    src/TMacOSXSystem.mm
  )

  target_compile_options(Macosx PRIVATE -ObjC++)
  target_include_directories(Macosx PRIVATE ../unix/inc)

  target_sources(Core PRIVATE $<TARGET_OBJECTS:Macosx>)

  if(NOT CMAKE_VERSION VERSION_LESS "3.23.0")
    target_sources(
        Macosx
        PRIVATE
        FILE_SET private_header_files
        TYPE HEADERS
        BASE_DIRS inc/
        FILES
            inc/CocoaUtils.h
            inc/TMacOSXSystem.h
            inc/LinkDef.h
    )
  endif()

endif()

ROOT_INSTALL_HEADERS()
