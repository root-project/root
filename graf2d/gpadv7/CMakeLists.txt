# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graf2d/gpadv7 package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTGpadv7
  HEADERS
    ROOT/RCanvas.hxx
    ROOT/RFrame.hxx
    ROOT/RMenuItem.hxx
    ROOT/RObjectDrawable.hxx
    ROOT/RColor.hxx
    ROOT/RDisplayItem.hxx
    ROOT/RDrawingAttr.hxx
    ROOT/RAttrBox.hxx
    ROOT/RAttrLine.hxx
    ROOT/RAttrMarker.hxx
    ROOT/RAttrText.hxx
    ROOT/RDrawingOptsBase.hxx
    ROOT/RPalette.hxx
    ROOT/RDrawable.hxx
    ROOT/RStyle.hxx
    ROOT/RPadDisplayItem.hxx
    ROOT/RPadExtent.hxx
    ROOT/RPad.hxx
    ROOT/RPadLength.hxx
    ROOT/RPadPainter.hxx
    ROOT/RPadPos.hxx
    ROOT/RPadUserAxis.hxx
    ROOT/RVirtualCanvasPainter.hxx
  SOURCES
    v7/src/RCanvas.cxx
    v7/src/RFrame.cxx
    v7/src/RMenuItem.cxx
    v7/src/RObjectDrawable.cxx
    v7/src/RColor.cxx
    v7/src/RDisplayItem.cxx
    v7/src/RDrawable.cxx
    v7/src/RDrawingAttr.cxx
    v7/src/RDrawingOptsBase.cxx
    v7/src/RPalette.cxx
    v7/src/RStyle.cxx
    v7/src/RStyleReader.cxx
    v7/src/RPad.cxx
    v7/src/RPadExtent.cxx
    v7/src/RPadLength.cxx
    v7/src/RPadPainter.cxx
    v7/src/RPadPos.cxx
    v7/src/RPadUserAxis.cxx
    v7/src/RVirtualCanvasPainter.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Core
    RIO
)


ROOT_ADD_TEST_SUBDIRECTORY(v7/test)
