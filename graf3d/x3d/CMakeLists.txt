# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graf3d/X3d C++ package
############################################################################

set(X3D_HEADERS
  TViewerX3D.h
  TX3DFrame.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(X3d
  HEADERS
    ${X3D_HEADERS}
  SOURCES
    src/TViewerX3D.cxx
    src/TX3DFrame.cxx
  DEPENDENCIES
    Graf3d
    Gui
  LIBRARIES
    ROOTX3dViewer
)

set(RELATIVE_INC_HEADERS ${X3D_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(X3d
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
  ${RELATIVE_INC_HEADERS}
  inc/LinkDef.h
)
