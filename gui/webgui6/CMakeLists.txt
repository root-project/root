# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/webgui6 package
############################################################################

set(WEBGUI_HEADERS
  TWebCanvas.h
  TWebControlBar.h
  TWebMenuItem.h
  TWebPadOptions.h
  TWebPadPainter.h
  TWebPainting.h
  TWebSnapshot.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(WebGui6
  HEADERS
    ${WEBGUI_HEADERS}
  SOURCES
    src/TWebCanvas.cxx
    src/TWebControlBar.cxx
    src/TWebMenuItem.cxx
    src/TWebPadPainter.cxx
    src/TWebPainting.cxx
    src/TWebSnapshot.cxx
  DEPENDENCIES
    Core
    Gpad
    RIO
    ROOTWebDisplay
    Thread
)

set(RELATIVE_INC_HEADERS ${WEBGUI_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(WebGui6
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)
