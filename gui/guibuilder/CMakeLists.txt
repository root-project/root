# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/guibuilder package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(GuiBld
  HEADERS
    TGuiBldDragManager.h
    TGuiBldEditor.h
    TGuiBldGeometryFrame.h
    TGuiBldHintsButton.h
    TGuiBldHintsEditor.h
    TGuiBldNameFrame.h
    TRootGuiBuilder.h
  SOURCES
    src/TGuiBldDragManager.cxx
    src/TGuiBldEditor.cxx
    src/TGuiBldGeometryFrame.cxx
    src/TGuiBldHintsButton.cxx
    src/TGuiBldHintsEditor.cxx
    src/TGuiBldNameFrame.cxx
    src/TRootGuiBuilder.cxx
  DEPENDENCIES
    Gui
    Graf
    MathCore
)
