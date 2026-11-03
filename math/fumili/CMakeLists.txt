# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT math/fumili package
############################################################################

set(FUMILI_HEADERS
  TFumili.h
  TFumiliMinimizer.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(Fumili
  HEADERS
    ${FUMILI_HEADERS}
  SOURCES
    src/TFumili.cxx
    src/TFumiliMinimizer.cxx
  DEPENDENCIES
    Graf
    Hist
    MathCore
)

set(RELATIVE_INC_HEADERS ${FUMILI_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(Fumili
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)
