# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/browserv7
############################################################################

ROOT_LINKER_LIBRARY(ROOTBrowserWidgets
     src/RBrowserWidget.cxx
  DEPENDENCIES
     ROOTBrowsable
     ROOTWebDisplay
)

ROOT_LINKER_LIBRARY(ROOTBrowserTCanvasWidget
     src/RBrowserTCanvasWidget.cxx
  DEPENDENCIES
     ROOTBrowserWidgets
     Gpad
     WebGui6
)

if(root7)
  ROOT_LINKER_LIBRARY(ROOTBrowserRCanvasWidget
       src/RBrowserRCanvasWidget.cxx
    DEPENDENCIES
       ROOTBrowserWidgets
       ROOTGpadv7
)
endif()

if(geom)
  ROOT_LINKER_LIBRARY(ROOTBrowserGeomWidget
       src/RBrowserGeomWidget.cxx
    DEPENDENCIES
       ROOTBrowserWidgets
       ROOTGeomViewer
  )
endif()

ROOT_LINKER_LIBRARY(ROOTBrowserTreeWidget
     src/RBrowserTreeWidget.cxx
  DEPENDENCIES
     ROOTBrowserWidgets
     ROOTBrowserv7
     ROOTTreeViewer
)

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTBrowserv7
  HEADERS
    ROOT/RBrowser.hxx
    ROOT/RBrowserData.hxx
    ROOT/RBrowserReply.hxx
    ROOT/RBrowserRequest.hxx
    ROOT/RFileDialog.hxx
    ROOT/RWebBrowserImp.hxx
  SOURCES
    src/RBrowser.cxx
    src/RBrowserData.cxx
    src/RFileDialog.cxx
    src/RWebBrowserImp.cxx
  LIBRARIES
    ROOTBrowserWidgets
  DEPENDENCIES
    ROOTBrowsable
    ROOTWebDisplay
    RIO
)


