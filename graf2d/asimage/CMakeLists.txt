# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graf2d/asimage package
# @author Pere Mato, CERN
############################################################################

add_subdirectory(src/libAfterImage)

ROOT_STANDARD_LIBRARY_PACKAGE(ASImage
  NO_INSTALL_HEADERS
  HEADERS
    TASImage.h
    TASImagePlugin.h
    TASPluginGS.h
    TASPngWriter.h
  SOURCES
    TASImage.cxx
    TASPluginGS.cxx
    TASPngWriter.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  LIBRARIES
    libAfterImage
    ${ASEXTRA_LIBRARIES}
    ${FREETYPE_LIBRARIES}
    ${X11_LIBRARIES}
    ZLIB::ZLIB
  DEPENDENCIES
    Core
    Graf
    Postscript
)

target_link_libraries(ASImage PRIVATE JPEG::JPEG PNG::PNG GIF::GIF libAfterImage Freetype::Freetype)
if (x11)
  target_link_libraries(ASImage PRIVATE X11::X11)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(ASImageGui
  NO_INSTALL_HEADERS
  HEADERS
    TASPaletteEditor.h
  SOURCES
    src/TASPaletteEditor.cxx
  LINKDEF
    LinkDefGui.h
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  LIBRARIES
    libAfterImage
    ${ASEXTRA_LIBRARIES}
    ${FREETYPE_LIBRARIES}
    ${X11_LIBRARIES}
    ZLIB::ZLIB
  DEPENDENCIES
    Gui
    ASImage
)

target_link_libraries(ASImageGui PRIVATE JPEG::JPEG PNG::PNG GIF::GIF libAfterImage Freetype::Freetype)
if (x11)
  target_link_libraries(ASImageGui PRIVATE X11::X11)
endif()

ROOT_INSTALL_HEADERS()
