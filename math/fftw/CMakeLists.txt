# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT math/fftw package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(FFTW
  HEADERS
    TFFTComplex.h
    TFFTComplexReal.h
    TFFTReal.h
    TFFTRealComplex.h
  SOURCES
    src/TFFTComplex.cxx
    src/TFFTComplexReal.cxx
    src/TFFTReal.cxx
    src/TFFTRealComplex.cxx
  DICTIONARY_OPTIONS
    -I${FFTW_INCLUDE_DIR}
  DEPENDENCIES
    Core
    MathCore
  BUILTINS
    FFTW3
)

target_include_directories(FFTW PRIVATE ${FFTW_INCLUDE_DIR})
target_link_libraries(FFTW PRIVATE ${FFTW_LIBRARIES})
