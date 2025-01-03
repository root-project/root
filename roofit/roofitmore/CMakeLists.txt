# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT roofitmore package
# @author Lorenzo Moneta, CERN
############################################################################

if(mathmore)
  set(ROOT_MATHMORE_LIBRARY MathMore)
endif()

set (EXTRA_DICT_OPTS)
if (runtime_cxxmodules AND WIN32)
  set (EXTRA_DICT_OPTS NO_CXXMODULE)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(RooFitMore
  HEADERS
    RooFitMoreLib.h
    RooLegendre.h
    RooMathMoreReg.h
    RooSpHarmonic.h
    RooNonCentralChiSquare.h
    RooHypatia2.h
  SOURCES
    src/RooFitMoreLib.cxx
    src/RooAdaptiveGaussKronrodIntegrator1D.cxx
    src/RooGaussKronrodIntegrator1D.cxx
    src/RooLegendre.cxx
    src/RooMathMoreReg.cxx
    src/RooSpHarmonic.cxx
    src/RooNonCentralChiSquare.cxx
    src/RooHypatia2.cxx
  DICTIONARY_OPTIONS
    "-writeEmptyRootPCM"
  LINKDEF
    LinkDef.h
  LIBRARIES
    RooBatchCompute
  DEPENDENCIES
    ${ROOT_MATHMORE_LIBRARY}
    Core
    RooFitCore
    RooFit
    Hist
    Matrix
    Tree
    Minuit
    RIO
    MathCore
    Foam
  BUILTINS
    GSL
  ${EXTRA_DICT_OPTS}
)

target_include_directories(RooFitMore SYSTEM PRIVATE ${GSL_INCLUDE_DIR})
target_link_libraries(RooFitMore PRIVATE ${GSL_LIBRARIES})

# For recent clang, this can facilitate auto-vectorisation.
# In RooFit, the errno side effect is not needed, anyway:
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  target_compile_options(RooFitMore PUBLIC -fno-math-errno)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
