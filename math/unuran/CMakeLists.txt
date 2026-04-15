# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT math/unuran package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Unuran
  HEADERS
    TUnuran.h
    TUnuranBaseDist.h
    TUnuranContDist.h
    TUnuranDiscrDist.h
    TUnuranEmpDist.h
    TUnuranMultiContDist.h
    TUnuranSampler.h
  SOURCES
    src/TUnuran.cxx
    src/TUnuranContDist.cxx
    src/TUnuranDiscrDist.cxx
    src/TUnuranEmpDist.cxx
    src/TUnuranMultiContDist.cxx
    src/TUnuranSampler.cxx
  LIBRARIES
    Unuran::unuran
  DEPENDENCIES
    Core
    Hist
    MathCore
)

ROOT_ADD_TEST_SUBDIRECTORY(test)
