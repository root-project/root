# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graf2d/x11ttf package
# @author Pere Mato, CERN
############################################################################

set(GX11TTF_HEADERS TGX11TTF.h)
ROOT_STANDARD_LIBRARY_PACKAGE(GX11TTF
  HEADERS
    ${GX11TTF_HEADERS}
  SOURCES
    src/TGX11TTF.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  LIBRARIES
    ${X11_LIBRARIES}
    ${X11_Xpm_LIB}
    ${X11_Xft_LIB}
    Freetype::Freetype
    ZLIB::ZLIB
    ${CMAKE_THREAD_LIBS_INIT}
  DEPENDENCIES
    GX11
    Graf
)

set(RELATIVE_INC_HEADERS ${GX11TTF_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(GX11TTF
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)

target_include_directories(GX11TTF PRIVATE ${X11_INCLUDE_DIR})
