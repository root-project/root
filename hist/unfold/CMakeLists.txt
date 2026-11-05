# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT hist/unfold package
############################################################################

set(UNFOLD_HEADERS
  TUnfold.h
  TUnfoldBinning.h
  TUnfoldBinningXML.h
  TUnfoldDensity.h
  TUnfoldIterativeEM.h
  TUnfoldSys.h
)

set(RELATIVE_INC_HEADERS ${UNFOLD_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)

ROOT_STANDARD_LIBRARY_PACKAGE(Unfold
  HEADERS
    ${UNFOLD_HEADERS}
  SOURCES
    src/TUnfold.cxx
    src/TUnfoldBinning.cxx
    src/TUnfoldBinningXML.cxx
    src/TUnfoldDensity.cxx
    src/TUnfoldIterativeEM.cxx
    src/TUnfoldSys.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Hist
    XMLParser
    Matrix
)

target_sources(Unfold
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)
