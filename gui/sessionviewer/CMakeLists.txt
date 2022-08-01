# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/sessionviewer package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(SessionViewer
  HEADERS
    TProofProgressDialog.h
    TProofProgressLog.h
    TProofProgressMemoryPlot.h
    TSessionDialogs.h
    TSessionLogView.h
    TSessionViewer.h
  SOURCES
    src/TProofProgressDialog.cxx
    src/TProofProgressLog.cxx
    src/TProofProgressMemoryPlot.cxx
    src/TSessionDialogs.cxx
    src/TSessionLogView.cxx
    src/TSessionViewer.cxx
  DEPENDENCIES
    Proof
    ROOTGui
    Hist
    Gpad
    Graf
    Tree
    MathCore
)
