# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT math/mlp package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(MLP
  HEADERS
    TMLPAnalyzer.h
    TMultiLayerPerceptron.h
    TNeuron.h
    TSynapse.h
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
