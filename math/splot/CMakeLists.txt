# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT math/splot package
############################################################################

set(SPLOT_HEADERS
  TSPlot.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(SPlot
  HEADERS
    ${SPLOT_HEADERS}
  SOURCES
    src/TSPlot.cxx
  DEPENDENCIES
    Matrix
    Hist
    Tree
    TreePlayer
    Graf
    MathCore
)

set(RELATIVE_INC_HEADERS ${SPLOT_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(SPlot
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)
