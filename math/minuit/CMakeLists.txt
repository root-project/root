# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT math/minuit package
############################################################################

set(MINUIT_HEADERS
  TFitter.h
  TLinearFitter.h
  TLinearMinimizer.h
  TMinuit.h
  TMinuitMinimizer.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(Minuit
  HEADERS
    ${MINUIT_HEADERS}
 SOURCES
    src/TFitter.cxx
    src/TLinearFitter.cxx
    src/TLinearMinimizer.cxx
    src/TMinuit.cxx
    src/TMinuitMinimizer.cxx
  DEPENDENCIES
    Graf
    Hist
    Matrix
    MathCore
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
)

set(RELATIVE_INC_HEADERS ${MINUIT_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(Minuit
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)

if (CMAKE_BUILD_TYPE STREQUAL Optimized)
  target_compile_options(Minuit PRIVATE "-fno-fast-math")
endif()
