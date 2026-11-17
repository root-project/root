# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT roofitmore package
# @author Lorenzo Moneta, CERN
############################################################################

add_library(RooFitMore SHARED)
target_sources(RooFitMore
PRIVATE
  src/RooFitMoreLib.cxx
  src/RooAdaptiveGaussKronrodIntegrator1D.cxx
  src/RooGaussKronrodIntegrator1D.cxx
  src/RooLegendre.cxx
  src/RooMathMoreReg.cxx
  src/RooSpHarmonic.cxx
  src/RooNonCentralChiSquare.cxx
  src/RooHypatia2.cxx
PRIVATE
  FILE_SET private_header_files TYPE HEADERS BASE_DIRS src/ FILES
    src/BracketAdapters.h
    src/RooAdaptiveGaussKronrodIntegrator1D.h
    src/RooGaussKronrodIntegrator1D.h
PUBLIC
  FILE_SET HEADERS TYPE HEADERS BASE_DIRS inc/ FILES
    inc/RooFitMoreLib.h
    inc/RooLegendre.h
    inc/RooMathMoreReg.h
    inc/RooSpHarmonic.h
    inc/RooNonCentralChiSquare.h
    inc/RooHypatia2.h
  FILE_SET linkDef TYPE HEADERS BASE_DIRS inc/ FILES
    inc/LinkDef.h
)

target_link_libraries(RooFitMore
PUBLIC
  Core
  RooFitCore
  RooFit
  Hist
  Tree
  RIO
PRIVATE
  MathCore
  GSL::gsl GSL::gslcblas
)
if(mathmore)
  target_link_libraries(RooFitMore PRIVATE MathMore)
endif()

set (EXTRA_DICT_OPTS)
if (runtime_cxxmodules AND WIN32)
  set (EXTRA_DICT_OPTS NO_CXXMODULE)
endif()

ROOT_GENERATE_TARGET_DICTIONARY(RooFitMore "-writeEmptyRootPCM" ${EXTRA_DICT_OPTS})
ROOT_INSTALL_TARGET(RooFitMore)

# For recent clang, this can facilitate auto-vectorisation.
# In RooFit, the errno side effect is not needed, anyway:
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  target_compile_options(RooFitMore PUBLIC -fno-math-errno)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
