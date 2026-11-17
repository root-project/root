# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graf2d/x11 package
# @author Pere Mato, CERN
############################################################################

set(GX11_HEADERS TGX11.h)
ROOT_STANDARD_LIBRARY_PACKAGE(GX11
  HEADERS
    ${GX11_HEADERS}
  SOURCES
    ../gifencode/gifdecode.cxx
    ../gifencode/gifencode.cxx
    src/GX11Gui.cxx
    src/TGX11.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  LIBRARIES
    ${X11_LIBRARIES}
    ${X11_Xpm_LIB}
    ${CMAKE_THREAD_LIBS_INIT}
    xvertext
  DEPENDENCIES
    Core
    MathCore
  INSTALL_OPTIONS
    FILTER "inc/X11/"
    FILTER "X11$"
)

set(RELATIVE_INC_HEADERS ${GX11_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(GX11
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/ ../gifencode
  FILES
    ${RELATIVE_INC_HEADERS}
    ../gifencode/gifencode.h
    ../gifencode/gifdecode.h
    inc/LinkDef.h
)

target_include_directories(GX11 PRIVATE ${X11_INCLUDE_DIR})
target_include_directories(GX11 PRIVATE ../gifencode)
