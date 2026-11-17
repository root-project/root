# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graf3d/gviz3d package
############################################################################

set (EXTRA_DICT_OPTS)
if (runtime_cxxmodules AND WIN32)
  set (EXTRA_DICT_OPTS NO_CXXMODULE)
endif()

set(GVIZ3D_HEADERS
  TStructNode.h
  TStructNodeEditor.h
  TStructNodeProperty.h
  TStructViewer.h
  TStructViewerGUI.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(Gviz3d
  HEADERS
    ${GVIZ3D_HEADERS}
  SOURCES
    src/TStructNode.cxx
    src/TStructNodeEditor.cxx
    src/TStructNodeProperty.cxx
    src/TStructViewer.cxx
    src/TStructViewerGUI.cxx
  LIBRARIES
    ${FTGL_LIBRARIES}
  DEPENDENCIES
    Gui
    Ged
    Geom
    RGL
   ${EXTRA_DICT_OPTS}
)

set(RELATIVE_INC_HEADERS ${GVIZ3D_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(Gviz3d
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
  ${RELATIVE_INC_HEADERS}
  inc/LinkDef.h
)
