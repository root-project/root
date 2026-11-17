# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT math/foam package
############################################################################

set(FOAM_HEADERS
  TFoam.h
  TFoamCell.h
  TFoamIntegrand.h
  TFoamMaxwt.h
  TFoamSampler.h
  TFoamVect.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(Foam
  HEADERS
    ${FOAM_HEADERS}
  SOURCES
    src/TFoam.cxx
    src/TFoamCell.cxx
    src/TFoamIntegrand.cxx
    src/TFoamMaxwt.cxx
    src/TFoamSampler.cxx
    src/TFoamVect.cxx
  DEPENDENCIES
    Hist
    MathCore
)

set(RELATIVE_INC_HEADERS ${FOAM_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(Foam
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)

ROOT_ADD_TEST_SUBDIRECTORY(test)
