# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT roofitcore package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(RooFitMore
  HEADERS
    RooAdaptiveGaussKronrodIntegrator1D.h
    RooGaussKronrodIntegrator1D.h
  SOURCES
    src/RooAdaptiveGaussKronrodIntegrator1D.cxx
    src/RooGaussKronrodIntegrator1D.cxx
  DICTIONARY_OPTIONS
    "-writeEmptyRootPCM"
  LINKDEF
    LinkDef1.h
  DEPENDENCIES
    ${ROOT_MATHMORE_LIBRARY}
    Core
    RooFitCore
    Hist
    Matrix
    Tree
    Minuit
    RIO
    MathCore
    Foam
)

#if(vdt AND NOT builtin_vdt)
#  target_include_directories(RooFitCore PRIVATE ${VDT_INCLUDE_DIR})
#endif()

#ROOT_ADD_TEST_SUBDIRECTORY(test)
