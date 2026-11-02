# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graf2d/cocoa package
# @author Pere Mato, CERN
############################################################################

find_library(COCOA_FRAMEWORK Cocoa REQUIRED)
find_library(OPENGL_FRAMEWORK OpenGL REQUIRED)

set(GCOCOA_HEADERS
  CocoaGuiTypes.h
  TGCocoa.h
  TGOSXGL.h
  TGQuartz.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(GCocoa
  HEADERS
    ${GCOCOA_HEADERS}
  SOURCES
    src/CocoaConstants.mm
    src/CocoaGuiTypes.mm
    src/CocoaPrivate.mm
    src/FontCache.mm
    src/MenuLoader.mm
    src/QuartzPixmap.mm
    src/QuartzWindow.mm
    src/ROOTApplicationDelegate.mm
    src/ROOTOpenGLView.mm
    src/TGCocoa.mm
    src/TGOSXGL.mm
    src/TGQuartz.mm
    src/X11Atoms.mm
    src/X11Buffer.mm
    src/X11Colors.mm
    src/X11Events.mm
    src/XLFDParser.mm
  DEPENDENCIES
    Gui
  LIBRARIES
    GQuartz
    ${COCOA_FRAMEWORK}
    ${OPENGL_FRAMEWORK}
    Freetype::Freetype
)

set(RELATIVE_INC_HEADERS ${GCOCOA_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(GCocoa
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/ src/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/CocoaConstants.h
    inc/FontCache.h
    inc/MenuLoader.h
    inc/QuartzPixmap.h
    inc/QuartzWindow.h
    inc/ROOTApplicationDelegate.h
    inc/ROOTOpenGLView.h
    inc/X11Atoms.h
    inc/X11Buffer.h
    inc/X11Colors.h
    inc/X11Drawable.h
    inc/X11Events.h
    inc/XLFDParser.h
    inc/LinkDef.h
    src/CocoaPrivate.h
)

target_include_directories(GCocoa PRIVATE ${CMAKE_SOURCE_DIR}/graf3d/gl/inc)

# We know GL is deprecated on macOS. And our answer is WebGUI/Graphics.
target_compile_options(GCocoa PRIVATE -ObjC++ -Wno-deprecated-declarations)
target_include_directories(GCocoa PRIVATE ${OPENGL_INCLUDE_DIR})
