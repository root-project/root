# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT hist/histpainter package
############################################################################

if(root7)
    set(HISTPAINTER_V7_SOURCES v7/src/RHistPainter.cxx)
    set(HISTPAINTER_V7_LIBRARIES ROOTHistDraw)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(HistPainter
  HEADERS
    Hoption.h
    Hparam.h
    TGraph2DPainter.h
    TGraphPainter.h
    THistPainter.h
    TLego.h
    TPainter3dAlgorithms.h
    TPaletteAxis.h
  SOURCES
    src/TGraph2DPainter.cxx
    src/TGraphPainter.cxx
    src/THistPainter.cxx
    src/TPainter3dAlgorithms.cxx
    src/TPaletteAxis.cxx
    ${HISTPAINTER_V7_SOURCES}
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Gpad
    Graf
    Hist
    MathCore
    Matrix
    ${HISTPAINTER_V7_LIBRARIES}
)
