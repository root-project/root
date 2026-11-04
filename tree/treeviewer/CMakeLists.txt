# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT tree/treeviewer package
# @author Pere Mato, CERN
############################################################################

set(TREEVIEWER_HEADERS
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
)

ROOT_STANDARD_LIBRARY_PACKAGE(TreeViewer
  HEADERS
    ${TREEVIEWER_HEADERS}
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
if(ROOT_NEED_STDCXXFS)
  target_link_libraries(TreeViewer PUBLIC stdc++fs)
endif()
# Suppress deprecation warnings across compilation units and during dictionary generation, remove with HelpTextTV.h
target_compile_definitions(TreeViewer PRIVATE ROOT_HelpTextTV_cxx)

set(RELATIVE_INC_HEADERS ${TREEVIEWER_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(TreeViewer
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)
