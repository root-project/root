# Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT treemap package
# @author Patryk Tymoteusz Pilichowski CERN
############################################################################

set(TREEMAP_HEADERS
  ROOT/RTreeMapBase.hxx
  ROOT/RTreeMapPainter.hxx
)

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTTreeMap
  HEADERS
    ${TREEMAP_HEADERS}
  SOURCES
    RTreeMapBase.cxx
    RTreeMapPainter.cxx
  DEPENDENCIES
    Gpad
    RIO
)

set(RELATIVE_INC_HEADERS ${TREEMAP_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(ROOTTreeMap
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)
