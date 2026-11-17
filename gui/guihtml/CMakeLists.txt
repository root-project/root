# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/guihtml package
############################################################################

set(GUIHTML_HEADERS
  TGHtml.h
  TGHtmlTokens.h
  TGHtmlUri.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(GuiHtml
  HEADERS
    ${GUIHTML_HEADERS}
  SOURCES
    src/TGHtml.cxx
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
    Gui
    Graf
    ASImageUtils
)

set(RELATIVE_INC_HEADERS ${GUIHTML_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(GuiHtml
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)
