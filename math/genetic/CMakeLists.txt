# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT math/genetic package
# @author Pere Mato, CERN
############################################################################

set(GENETIC_HEADERS
  Math/GeneticMinimizer.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(Genetic
  HEADERS
    ${GENETIC_HEADERS}
  SOURCES
    src/GeneticMinimizer.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Core
    MathCore
    TMVA
)

set(RELATIVE_INC_HEADERS ${GENETIC_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(Genetic
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)

ROOT_ADD_TEST_SUBDIRECTORY(test)
