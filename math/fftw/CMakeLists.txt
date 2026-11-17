# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT math/fftw package
# @author Pere Mato, CERN
############################################################################

set(FFTW_HEADERS
  TFFTComplex.h
  TFFTComplexReal.h
  TFFTReal.h
  TFFTRealComplex.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(FFTW
  HEADERS
    ${FFTW_HEADERS}
  SOURCES
    src/TFFTComplex.cxx
    src/TFFTComplexReal.cxx
    src/TFFTReal.cxx
    src/TFFTRealComplex.cxx
  LIBRARIES
    FFTW::Double
  DEPENDENCIES
    Core
    MathCore
)

set(RELATIVE_INC_HEADERS ${FFTW_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(FFTW
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)
