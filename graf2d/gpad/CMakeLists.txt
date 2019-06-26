# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graf2d/gpad package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Gpad
  HEADERS
    TAttCanvas.h
    TButton.h
    TCanvas.h
    TClassTree.h
    TColorWheel.h
    TControlBarButton.h
    TControlBar.h
    TCreatePrimitives.h
    TDialogCanvas.h
    TGroupButton.h
    TInspectCanvas.h
    TPad.h
    TPadPainter.h
    TPaveClass.h
    TRatioPlot.h
    TSliderBox.h
    TSlider.h
    TViewer3DPad.h
    TView.h
  SOURCES
    src/TAttCanvas.cxx
    src/TButton.cxx
    src/TCanvas.cxx
    src/TClassTree.cxx
    src/TColorWheel.cxx
    src/TControlBarButton.cxx
    src/TControlBar.cxx
    src/TCreatePrimitives.cxx
    src/TDialogCanvas.cxx
    src/TGroupButton.cxx
    src/TInspectCanvas.cxx
    src/TPad.cxx
    src/TPadPainter.cxx
    src/TPaveClass.cxx
    src/TRatioPlot.cxx
    src/TSliderBox.cxx
    src/TSlider.cxx
    src/TView.cxx
    src/TViewer3DPad.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Graf
    Hist
)
