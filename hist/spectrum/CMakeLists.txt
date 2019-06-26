# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT hist/spectrum package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Spectrum
  HEADERS
    TSpectrum.h
    TSpectrum2.h
    TSpectrum2Fit.h
    TSpectrum2Transform.h
    TSpectrum3.h
    TSpectrumFit.h
    TSpectrumTransform.h
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
    Matrix
)
