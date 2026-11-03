# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT math/mlp package
############################################################################

set(MLP_HEADERS
  TMLPAnalyzer.h
  TMultiLayerPerceptron.h
  TNeuron.h
  TSynapse.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(MLP
  HEADERS
    ${MLP_HEADERS}
  SOURCES
    src/TMLPAnalyzer.cxx
    src/TMultiLayerPerceptron.cxx
    src/TNeuron.cxx
    src/TSynapse.cxx
  DEPENDENCIES
    Hist
    Matrix
    Tree
    Graf
    Gpad
    TreePlayer
    MathCore
)

set(RELATIVE_INC_HEADERS ${MLP_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(MLP
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)
