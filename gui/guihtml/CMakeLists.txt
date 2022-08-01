# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/guihtml package
############################################################################

if(ssl)
  add_definitions(-DR__SSL)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTGuiHtml
  BACKWARDS_COMPAT GuiHtml
  HEADERS
    TGHtml.h
    TGHtmlBrowser.h
    TGHtmlTokens.h
    TGHtmlUri.h
  SOURCES
    src/TGHtml.cxx
    src/TGHtmlBrowser.cxx
    src/TGHtmlDraw.cxx
    src/TGHtmlElement.cxx
    src/TGHtmlForm.cxx
    src/TGHtmlImage.cxx
    src/TGHtmlIndex.cxx
    src/TGHtmlLayout.cxx
    src/TGHtmlParse.cxx
    src/TGHtmlSizer.cxx
    src/TGHtmlTable.cxx
    src/TGHtmlTokenMap.cxx
    src/TGHtmlUri.cxx
  DEPENDENCIES
    ROOTGui
    Graf
    Net
)
