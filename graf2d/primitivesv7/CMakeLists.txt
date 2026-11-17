# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graf2d/primitives package
# @author Pere Mato, CERN
############################################################################

set(GRAPHICSPRIMITIVES_HEADERS
  ROOT/RBox.hxx
  ROOT/RFont.hxx
  ROOT/RFrameTitle.hxx
  ROOT/RLegend.hxx
  ROOT/RLine.hxx
  ROOT/RMarker.hxx
  ROOT/RPaveText.hxx
  ROOT/RText.hxx
)

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTGraphicsPrimitives
  HEADERS
    ${GRAPHICSPRIMITIVES_HEADERS}
  SOURCES
    src/RBox.cxx
    src/RFont.cxx
    src/RLegend.cxx
    src/RLine.cxx
    src/RMarker.cxx
    src/RText.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Core
    ROOTGpadv7
)

set(RELATIVE_INC_HEADERS ${GRAPHICSPRIMITIVES_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(ROOTGraphicsPrimitives
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)

ROOT_ADD_TEST_SUBDIRECTORY(test)
