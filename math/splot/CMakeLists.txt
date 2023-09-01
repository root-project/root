# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT math/splot package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(SPlot
  HEADERS
    TSPlot.h
  SOURCES
    src/TSPlot.cxx
  DEPENDENCIES
    Matrix
    Hist
    Tree
    TreePlayer
    Graf3d
    Graf
    MathCore
)
