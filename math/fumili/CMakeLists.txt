# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT math/fumili package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Fumili
  HEADERS
    TFumili.h
    TFumiliMinimizer.h
  SOURCES
    src/TFumili.cxx
    src/TFumiliMinimizer.cxx
  DEPENDENCIES
    Graf
    Hist
    MathCore
)
