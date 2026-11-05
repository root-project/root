# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

set(WEBTREEVIEWER_HEADERS
  ROOT/RTreeViewer.hxx
)

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTTreeViewer
  HEADERS
    ${WEBTREEVIEWER_HEADERS}
  SOURCES
    src/RTreeViewer.cxx
  DEPENDENCIES
    Tree
    Hist
    ROOTWebDisplay
)

set(RELATIVE_INC_HEADERS ${WEBTREEVIEWER_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(ROOTTreeViewer
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)
