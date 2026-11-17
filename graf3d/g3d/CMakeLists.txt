# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT graf3d/g3d package
############################################################################

set(GRAF3D_HEADERS
  TAxis3D.h
  TGeometryTransformer.h
  TMarker3DBox.h
  TPoints3DABC.h
  TPointSet3D.h
  TPolyLine3D.h
  TPolyMarker3D.h
  TView3D.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(Graf3d
  HEADERS
    ${GRAF3D_HEADERS}
  SOURCES
    TAxis3D.cxx
    TMarker3DBox.cxx
    TPoints3DABC.cxx
    TPointSet3D.cxx
    TPolyLine3D.cxx
    TPolyMarker3D.cxx
    TView3D.cxx
  DEPENDENCIES
    Gpad
    Graf
    Hist
    MathCore
  LIBRARIES
    ROOTX3dViewer
)

set(RELATIVE_INC_HEADERS ${GRAF3D_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(Graf3d
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
  ${RELATIVE_INC_HEADERS}
  inc/LinkDef.h
)
