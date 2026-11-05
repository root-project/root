# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT hist/spectrum package
############################################################################

set(SPECTRUM_HEADERS
  TSpectrum.h
  TSpectrum2.h
  TSpectrum2Fit.h
  TSpectrum2Transform.h
  TSpectrum3.h
  TSpectrumFit.h
  TSpectrumTransform.h
)

set(RELATIVE_INC_HEADERS ${SPECTRUM_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)

ROOT_STANDARD_LIBRARY_PACKAGE(Spectrum
  HEADERS
    ${SPECTRUM_HEADERS}
  SOURCES
    src/TSpectrum.cxx
    src/TSpectrum2.cxx
    src/TSpectrum2Fit.cxx
    src/TSpectrum2Transform.cxx
    src/TSpectrum3.cxx
    src/TSpectrumFit.cxx
    src/TSpectrumTransform.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Hist
)

target_sources(Spectrum
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)
