# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT hist/histpainter package
############################################################################

set(HISTPAINTER_HEADERS
  Hoption.h
  Hparam.h
  TGraph2DPainter.h
  TGraphPainter.h
  THistPainter.h
  TPaletteAxis.h
)

set(RELATIVE_INC_HEADERS ${HISTPAINTER_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)

ROOT_STANDARD_LIBRARY_PACKAGE(HistPainter
  HEADERS
    ${HISTPAINTER_HEADERS}
  SOURCES
    src/TGraph2DPainter.cxx
    src/TGraphPainter.cxx
    src/THistPainter.cxx
    src/TPainter3dAlgorithms.cxx
    src/TPaletteAxis.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Gpad
    Graf
    Hist
    MathCore
  LIBRARIES
    Matrix
)
# Suppress deprecation warnings across compilation units and during dictionary generation, remove with Hoption.h Hparam.h
target_compile_definitions(HistPainter PRIVATE ROOT_Hoption_cxx ROOT_Hparam_cxx)

target_sources(HistPainter
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/ src/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
    src/TPainter3dAlgorithms.h
)

