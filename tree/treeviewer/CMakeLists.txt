# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT tree/treeviewer package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(TreeViewer
  HEADERS
    HelpTextTV.h
    TGTreeTable.h
    TParallelCoordEditor.h
    TParallelCoord.h
    TParallelCoordRange.h
    TParallelCoordVar.h
    TSpiderEditor.h
    TSpider.h
    TTreeViewer.h
    TTVLVContainer.h
    TTVSession.h
  SOURCES
    src/HelpTextTV.cxx
    src/TGTreeTable.cxx
    src/TParallelCoord.cxx
    src/TParallelCoordEditor.cxx
    src/TParallelCoordRange.cxx
    src/TParallelCoordVar.cxx
    src/TSpider.cxx
    src/TSpiderEditor.cxx
    src/TTreeViewer.cxx
    src/TTVLVContainer.cxx
    src/TTVSession.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Ged
    Gui
    TreePlayer
)
if (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(TreeViewer PUBLIC stdc++fs) # https://github.com/avast/retdec/issues/842
endif()

