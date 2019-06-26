# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT math/rtools package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Rtools
  HEADERS
    Math/RMinimizer.h
  SOURCES
    src/RMinimizer.cxx
  DEPENDENCIES
    Core
    MathCore
    RInterface
)

file(COPY "${CMAKE_SOURCE_DIR}/etc/plugins/ROOT@@Math@@Minimizer/P090_RMinimizer.C"
  DESTINATION "${CMAKE_BINARY_DIR}/etc/plugins/ROOT@@Math@@Minimizer/")
