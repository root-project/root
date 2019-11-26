# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/ged package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Ged
  HEADERS
    HelpSMText.h
    TArrowEditor.h
    TAttFillEditor.h
    TAttLineEditor.h
    TAttMarkerEditor.h
    TAttTextEditor.h
    TAxisEditor.h
    TCurlyArcEditor.h
    TCurlyLineEditor.h
    TF1Editor.h
    TFrameEditor.h
    TFunctionParametersDialog.h
    TGedEditor.h
    TGedFrame.h
    TGedMarkerSelect.h
    TGedPatternSelect.h
    TGraphEditor.h
    TH1Editor.h
    TH2Editor.h
    TLineEditor.h
    TPadEditor.h
    TPaveStatsEditor.h
    TPieEditor.h
    TPieSliceEditor.h
    TStyleDialog.h
    TStyleManager.h
    TStylePreview.h
    TTextEditor.h
  SOURCES
    src/HelpSMText.cxx
    src/TArrowEditor.cxx
    src/TAttFillEditor.cxx
    src/TAttLineEditor.cxx
    src/TAttMarkerEditor.cxx
    src/TAttTextEditor.cxx
    src/TAxisEditor.cxx
    src/TCurlyArcEditor.cxx
    src/TCurlyLineEditor.cxx
    src/TF1Editor.cxx
    src/TFrameEditor.cxx
    src/TFunctionParametersDialog.cxx
    src/TGedEditor.cxx
    src/TGedFrame.cxx
    src/TGedMarkerSelect.cxx
    src/TGedPatternSelect.cxx
    src/TGraphEditor.cxx
    src/TH1Editor.cxx
    src/TH2Editor.cxx
    src/TLineEditor.cxx
    src/TPadEditor.cxx
    src/TPaveStatsEditor.cxx
    src/TPieEditor.cxx
    src/TPieSliceEditor.cxx
    src/TStyleDialog.cxx
    src/TStyleManager.cxx
    src/TStylePreview.cxx
    src/TTextEditor.cxx
  DEPENDENCIES
    Hist
    Gpad
    Graf
    Gui
    Tree
    TreePlayer
)
