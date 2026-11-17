# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT hist/spectrumpainter package
############################################################################

set(SPECTRUMPAINTER_HEADERS
  TSpectrum2Painter.h
)

set(RELATIVE_INC_HEADERS ${SPECTRUMPAINTER_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)

ROOT_STANDARD_LIBRARY_PACKAGE(SpectrumPainter
  HEADERS
    ${SPECTRUMPAINTER_HEADERS}
  SOURCES
    src/TSpectrum2Painter.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Graf
    Hist
    Gpad
)

target_sources(SpectrumPainter
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)
